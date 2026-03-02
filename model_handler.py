"""
Model handler for Qwen3.5-9B.

Responsible for:
- Loading the model and tokenizer from Hugging Face.
- Formatting prompts using the model's built-in chat template.
- Extracting hidden-state embeddings from a configurable layer.
- Generating text while capturing per-token entropy for hallucination
  detection.

Qwen3.5-9B has 32 transformer layers and a hidden dimension of 4096.
It uses a hybrid Gated DeltaNet + Gated Attention architecture.
The recommended analysis range is layers 16-26 (deep semantic zone).

Architecture notes:
- 32 transformer layers (indices 0-31).
- hidden_states tuple has length 33: index 0 = initial embeddings,
  indices 1-32 = outputs of each transformer layer.
- Hybrid SSM+attention: past_key_values includes both KV cache (attention
  layers) and recurrent state (DeltaNet layers). HuggingFace handles this
  transparently via the unified past_key_values API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import AsyncGenerator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_LAYER_IDX = 20
DEFAULT_MAX_NEW_TOKENS = 100
# Qwen3.5-9B has 32 transformer layers (indices 0-31).
# hidden_states tuple has length 33: index 0 = initial embeddings,
# indices 1-32 = outputs of each transformer layer.
NUM_LAYERS = 32
LAYER_RANGE = list(range(12, 29))  # 12-28 inclusive for the UI dropdown


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Holds everything produced by a single generation call."""

    response_text: str
    prompt_embeddings: np.ndarray          # (prompt_tokens, hidden_dim)
    response_embeddings: np.ndarray        # (response_tokens, hidden_dim)
    full_embeddings: np.ndarray            # (all_tokens, hidden_dim)
    token_entropies: list[float]           # per generated token
    tokens: list[str]                      # decoded tokens (full sequence)
    prompt_tokens: list[str]
    response_tokens: list[str]
    mean_entropy: float = 0.0
    max_entropy: float = 0.0

    # Indices for splitting prompt vs response in the full sequence
    prompt_len: int = 0


@dataclass
class ConversationTurn:
    """One turn in a multi-turn conversation."""

    role: str                              # "baseline" | "challenge"
    prompt: str
    result: GenerationResult | None = None
    messages: list[dict] = field(default_factory=list)


@dataclass
class StreamCheckpoint:
    """Snapshot emitted every N tokens during streaming generation.

    ``prompt_embeddings`` is constant across all checkpoints for a given turn.
    ``response_embeddings`` grows by one row per processed token.

    The hidden state for token *k* comes from the forward pass where *k* is
    the *input* token — i.e. we know the embedding for each generated token
    only after the model has processed it and produced logits for *k+1*.
    This introduces a one-token lag: the final EOS token has no embedding.
    """

    token_idx: int                     # 0-based index of last embedded token
    tokens_so_far: list[str]          # raw BPE tokens (for entropy chart labels)
    response_text: str                # human-readable text decoded so far
    prompt_embeddings: np.ndarray     # (prompt_len, hidden_dim) — constant
    response_embeddings: np.ndarray   # (n_embedded, hidden_dim) — grows each cp
    token_entropies: list[float]      # Shannon entropy (nats) per response token
    is_eos: bool                      # True on the final checkpoint


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ModelHandler:
    """Wraps Qwen3.5-9B for embedding extraction and generation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        layer_idx: int = DEFAULT_LAYER_IDX,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForCausalLM | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and load the model + tokenizer."""
        if self._loaded:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Chat-template formatting
    # ------------------------------------------------------------------

    def format_prompt(self, messages: list[dict]) -> str:
        """Apply the model's chat template to a list of messages.

        ``messages`` follows the OpenAI-style format::

            [{"role": "user", "content": "..."}, ...]
        """
        assert self.tokenizer is not None, "Call load() first."
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    # Embedding extraction (forward pass only, no generation)
    # ------------------------------------------------------------------

    def extract_embeddings(
        self,
        text: str,
        layer_idx: int | None = None,
    ) -> np.ndarray:
        """Run a forward pass and return hidden states from *layer_idx*.

        Parameters
        ----------
        text:
            Already-formatted text (e.g. output of ``format_prompt``).
        layer_idx:
            Transformer layer to extract.  ``None`` uses the instance
            default.

        Returns
        -------
        numpy array of shape ``(seq_len, hidden_dim)``.
        """
        assert self.model is not None and self.tokenizer is not None
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (num_layers + 1) tensors
        # Index 0 = raw token embeddings, 1..N = layer outputs.
        hidden_state = outputs.hidden_states[layer_idx + 1]
        return hidden_state.squeeze(0).cpu().float().numpy()

    # ------------------------------------------------------------------
    # Generation with embedding + entropy capture
    # ------------------------------------------------------------------

    def generate_with_embeddings(
        self,
        messages: list[dict],
        layer_idx: int | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate a response and extract embeddings + per-token entropy.

        The function:
        1. Formats the messages with the chat template.
        2. Runs ``model.generate`` to produce the response.
        3. Runs a single forward pass over the *full* sequence
           (prompt + response) to obtain hidden states from the target
           layer.
        4. Computes per-token entropy from the logits of step 3.

        Returns a ``GenerationResult`` dataclass.
        """
        assert self.model is not None and self.tokenizer is not None
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx

        # 1. Format ---------------------------------------------------
        text = self.format_prompt(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        # 2. Generate -------------------------------------------------
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
            )

        generated_ids = gen_outputs.sequences  # (1, total_len)
        response_ids = generated_ids[0, prompt_len:]
        response_text = self.tokenizer.decode(
            response_ids, skip_special_tokens=True,
        )

        # 3. Forward pass over full sequence --------------------------
        with torch.no_grad():
            full_outputs = self.model(
                generated_ids, output_hidden_states=True,
            )

        hidden_state = full_outputs.hidden_states[layer_idx + 1]
        full_embeddings = hidden_state.squeeze(0).cpu().float().numpy()

        prompt_embeddings = full_embeddings[:prompt_len]
        response_embeddings = full_embeddings[prompt_len:]

        # 4. Per-token entropy ----------------------------------------
        # Logits shape: (1, total_len, vocab_size)
        logits = full_outputs.logits.squeeze(0).cpu().float()
        # We only care about the response portion.
        response_logits = logits[prompt_len - 1 : -1]  # shifted by one
        token_entropies = _compute_token_entropies(response_logits)

        # 5. Decode individual tokens ---------------------------------
        all_ids = generated_ids.squeeze(0).tolist()
        all_tokens = [self.tokenizer.decode([tid]) for tid in all_ids]
        prompt_tokens = all_tokens[:prompt_len]
        resp_tokens = all_tokens[prompt_len:]

        mean_ent = float(np.mean(token_entropies)) if token_entropies else 0.0
        max_ent = float(np.max(token_entropies)) if token_entropies else 0.0

        return GenerationResult(
            response_text=response_text,
            prompt_embeddings=prompt_embeddings,
            response_embeddings=response_embeddings,
            full_embeddings=full_embeddings,
            token_entropies=token_entropies,
            tokens=all_tokens,
            prompt_tokens=prompt_tokens,
            response_tokens=resp_tokens,
            mean_entropy=mean_ent,
            max_entropy=max_ent,
            prompt_len=prompt_len,
        )


    async def stream_tokens_async(
        self,
        messages: list[dict],
        layer_idx: int | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
        checkpoint_every: int = 5,
    ) -> AsyncGenerator[StreamCheckpoint, None]:
        """Async generator: yield :class:`StreamCheckpoint` every *checkpoint_every* tokens.

        Uses a manual KV-cache autoregressive loop so we capture per-token
        hidden states without a redundant forward pass.  The cost over plain
        ``model.generate`` is negligible on GPU.

        Algorithm
        ---------
        1. Full forward pass on the prompt → prompt embeddings + first-token
           logits + KV cache.
        2. For each response token:
           a. Sample token *t* from current logits.
           b. Forward pass with *t* as sole input (KV cache handles context) →
              hidden state for *t* + logits for *t+1*.
           c. Append hidden state and entropy.
           d. Yield a :class:`StreamCheckpoint` every *checkpoint_every* steps.
        3. Always yield a final checkpoint with ``is_eos=True``.
        """
        import asyncio

        assert self.model is not None and self.tokenizer is not None
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx
        eos_id = self.tokenizer.eos_token_id

        text = self.format_prompt(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]          # (1, prompt_len)
        hidden_dim = self.model.config.hidden_size

        # --- Prompt forward pass -------------------------------------------
        with torch.no_grad():
            init_out = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True,
            )
        prompt_embeddings = (
            init_out.hidden_states[layer_idx + 1].squeeze(0).cpu().float().numpy()
        )
        past_key_values = init_out.past_key_values
        next_logits = init_out.logits[:, -1, :]   # (1, vocab_size)

        generated_ids: list[int] = []
        response_hiddens: list[np.ndarray] = []
        token_entropies: list[float] = []
        is_eos = False

        def _make_checkpoint(final: bool) -> StreamCheckpoint:
            resp_ids = generated_ids[:-1] if (final and is_eos) else generated_ids
            resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            bpe_tokens = self.tokenizer.convert_ids_to_tokens(resp_ids) or []
            embs = (
                np.stack(response_hiddens)
                if response_hiddens
                else np.empty((0, hidden_dim), dtype=np.float32)
            )
            return StreamCheckpoint(
                token_idx=len(generated_ids) - 1,
                tokens_so_far=bpe_tokens,
                response_text=resp_text,
                prompt_embeddings=prompt_embeddings,
                response_embeddings=embs,
                token_entropies=list(token_entropies),
                is_eos=final,
            )

        steps_since_cp = 0

        for _step in range(max_new_tokens):
            # Yield to the event loop so marimo can process state updates.
            await asyncio.sleep(0)

            # Sample next token from current logits.
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            next_id = next_token.item()

            # Entropy for this prediction step.
            ep = torch.softmax(next_logits[0], dim=-1).clamp(min=1e-12)
            entropy = float(-(ep * ep.log()).sum())

            generated_ids.append(next_id)
            token_entropies.append(entropy)
            is_eos = next_id == eos_id

            if is_eos:
                break

            # Forward pass: get hidden state for the token we just sampled.
            with torch.no_grad():
                step_out = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
            h = step_out.hidden_states[layer_idx + 1][0, 0].cpu().float().numpy()
            response_hiddens.append(h)
            past_key_values = step_out.past_key_values
            next_logits = step_out.logits[:, -1, :]

            steps_since_cp += 1
            if steps_since_cp >= checkpoint_every:
                steps_since_cp = 0
                yield _make_checkpoint(final=False)

        # Always emit a terminal checkpoint.
        yield _make_checkpoint(final=True)


# ---------------------------------------------------------------------------
# Hallucination helpers
# ---------------------------------------------------------------------------

def _compute_token_entropies(logits: torch.Tensor) -> list[float]:
    """Compute Shannon entropy for each position's probability distribution.

    Parameters
    ----------
    logits:
        Tensor of shape ``(num_tokens, vocab_size)``.

    Returns
    -------
    List of entropy values (in nats) for each token.
    """
    if logits.numel() == 0:
        return []
    probs = torch.softmax(logits, dim=-1)
    # Clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-12))
    entropies = -(probs * log_probs).sum(dim=-1)
    return entropies.tolist()


def detect_hallucination(
    result: GenerationResult,
    entropy_threshold: float = 4.0,
) -> tuple[bool, float, list[int]]:
    """Simple hallucination detector based on per-token entropy.

    A token whose entropy exceeds *entropy_threshold* is flagged.  If more
    than 30 % of response tokens are flagged the response is considered a
    potential hallucination.

    Returns
    -------
    (is_hallucination, confidence, flagged_token_indices)
    """
    if not result.token_entropies:
        return False, 0.0, []

    flagged = [
        i for i, e in enumerate(result.token_entropies)
        if e > entropy_threshold
    ]
    ratio = len(flagged) / len(result.token_entropies)
    is_hallucination = ratio > 0.30
    confidence = min(ratio / 0.60, 1.0)  # linearly scale to 1.0
    return is_hallucination, confidence, flagged


def compute_embedding_consistency(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> float:
    """Cosine similarity between the mean embeddings of two turns.

    Values close to 1.0 indicate the model's internal representation is
    consistent; values close to 0.0 suggest a large semantic shift.
    """
    mean_a = emb_a.mean(axis=0)
    mean_b = emb_b.mean(axis=0)
    dot = np.dot(mean_a, mean_b)
    norm = np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
    if norm < 1e-12:
        return 0.0
    return float(dot / norm)
