"""
Model handler for Qwen3 language models.

Handles model loading, text generation with hidden state extraction,
per-token entropy computation, and embedding capture for TDA analysis.

Supports both CPU and GPU inference with automatic device detection.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Default model - Qwen3-4B is the closest to 3B in the Qwen3 family
DEFAULT_MODEL = "Qwen/Qwen3-4B"


@dataclass
class GenerationResult:
    """Container for a single generation result with all analysis data."""

    prompt: str
    response: str
    embeddings: np.ndarray  # (num_tokens, hidden_dim) from the chosen layer
    prompt_embeddings: np.ndarray  # embeddings for prompt tokens only
    response_embeddings: np.ndarray  # embeddings for response tokens only
    token_entropies: list[float]  # per-token entropy during generation
    tokens: list[str]  # decoded tokens for hover labels
    prompt_tokens: list[str]
    response_tokens: list[str]
    mean_entropy: float
    max_entropy: float
    perplexity: float
    layer_idx: int
    generation_time: float


class ModelHandler:
    """Handles Qwen3 model loading, inference, and embedding extraction."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self.num_layers = None
        self.hidden_dim = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def get_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self, model_name: str = DEFAULT_MODEL) -> str:
        """
        Load a Qwen3 model and tokenizer.

        Returns a status message describing what was loaded.
        """
        self.device = self.get_device()
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16

        logger.info(f"Loading {model_name} on {self.device} with {dtype}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device.type,
            trust_remote_code=True,
        )
        self.model.eval()

        # Read architecture details from the model config
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        self.model_name = model_name

        status = (
            f"Loaded {model_name}\n"
            f"Device: {self.device} | Dtype: {dtype}\n"
            f"Layers: {self.num_layers} | Hidden dim: {self.hidden_dim}"
        )
        logger.info(status)
        return status

    def get_layer_range(self) -> tuple[int, int]:
        """
        Return the recommended layer range for the 'deep semantic zone'.

        This is the middle-to-upper portion of the model where embeddings
        have moved past basic syntax but haven't collapsed into vocabulary
        predictions. Roughly layers 50%-78% of the model depth.
        """
        if self.num_layers is None:
            return (14, 22)
        start = self.num_layers // 2
        end = int(self.num_layers * 0.78)
        return (start, end)

    def get_default_layer(self) -> int:
        """Return a good default layer index (middle of semantic zone)."""
        start, end = self.get_layer_range()
        return (start + end) // 2

    def generate_with_analysis(
        self,
        prompt: str,
        conversation_history: list[dict] | None = None,
        layer_idx: int | None = None,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """
        Generate a response and extract embeddings + entropy data.

        This is the core function: it runs the prompt through the model,
        captures the response, then does a forward pass on the full
        sequence (prompt + response) to extract hidden-state embeddings
        from the requested layer.

        Args:
            prompt: The user message to send.
            conversation_history: Prior messages as [{"role":..,"content":..}].
            layer_idx: Which transformer layer to extract (0-indexed).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            GenerationResult with embeddings, entropy, and text data.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if layer_idx is None:
            layer_idx = self.get_default_layer()

        start_time = time.time()

        # Build the message list with conversation history
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        # Format using the model's chat template
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: disable thinking mode
            )
        except TypeError:
            # Fallback if enable_thinking isn't supported
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]

        # --- Generate the response ---
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,  # needed for per-token entropy
            )

        full_ids = gen_outputs.sequences  # (1, prompt_len + gen_len)
        gen_ids = full_ids[0, prompt_length:]

        # Decode the response, stripping special / thinking tokens
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Strip any residual <think>...</think> blocks from Qwen3
        if "<think>" in response_text:
            import re

            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            ).strip()

        # --- Per-token entropy from generation scores ---
        token_entropies = []
        if gen_outputs.scores:
            for score in gen_outputs.scores:
                probs = torch.softmax(score[0], dim=-1)
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum().item()
                token_entropies.append(entropy)

        # --- Forward pass on full sequence to extract hidden states ---
        with torch.no_grad():
            outputs = self.model(full_ids, output_hidden_states=True)

        # hidden_states is a tuple: index 0 = embedding layer, 1..N = transformer layers
        # So transformer layer `layer_idx` (0-based) is at index `layer_idx + 1`
        hidden_state = outputs.hidden_states[layer_idx + 1]
        all_embeddings = hidden_state.squeeze(0).cpu().float().numpy()

        prompt_embeddings = all_embeddings[:prompt_length]
        response_embeddings = all_embeddings[prompt_length:]

        # --- Decode individual tokens for hover labels ---
        all_token_ids = full_ids[0].tolist()
        all_tokens = [self.tokenizer.decode([tid]) for tid in all_token_ids]
        prompt_tok_labels = all_tokens[:prompt_length]
        response_tok_labels = all_tokens[prompt_length:]

        generation_time = time.time() - start_time

        mean_ent = float(np.mean(token_entropies)) if token_entropies else 0.0
        max_ent = float(np.max(token_entropies)) if token_entropies else 0.0
        perplexity = float(np.exp(mean_ent)) if mean_ent > 0 else 1.0

        return GenerationResult(
            prompt=prompt,
            response=response_text,
            embeddings=all_embeddings,
            prompt_embeddings=prompt_embeddings,
            response_embeddings=response_embeddings,
            token_entropies=token_entropies,
            tokens=all_tokens,
            prompt_tokens=prompt_tok_labels,
            response_tokens=response_tok_labels,
            mean_entropy=mean_ent,
            max_entropy=max_ent,
            perplexity=perplexity,
            layer_idx=layer_idx,
            generation_time=generation_time,
        )

    def extract_embeddings_only(
        self, text: str, layer_idx: int | None = None
    ) -> np.ndarray:
        """
        Run a forward pass and return embeddings from a single layer.
        Useful for quick comparisons without full generation.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        if layer_idx is None:
            layer_idx = self.get_default_layer()

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_state = outputs.hidden_states[layer_idx + 1]
        return hidden_state.squeeze(0).cpu().float().numpy()
