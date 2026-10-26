"""Shared T5 generation wrapper (Layer 0 of the HealGen pipeline).

Every system (baseline, RAG, self-check, HealGen) generates answers through
this module so the conditioning method is the only variable between them:
greedy decoding (``do_sample=False``, ``num_beams=1``) with at most
``config.MAX_NEW_TOKENS`` new tokens.  ``torch.manual_seed(0)`` is set at
init and the model is kept in eval mode, so two calls with the same prompt
return the same string.

``get_generator(smoke=True)`` loads the small smoke model for demos and
tests; ``get_generator()`` loads the pinned experiment model for real runs.
Singletons are cached per model id so repeated calls do not reload weights.
``.n_tokens`` counts generated tokens (excluding the decoder start token)
for CPU throughput measurement.
"""

import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from healgen import config

_LOCK = threading.Lock()
_GENERATORS: dict[str, "Generator"] = {}


class Generator:
    """Greedy seq2seq wrapper around a Hugging Face checkpoint."""

    def __init__(self, model_id: str, smoke: bool = False):
        self.model_id = model_id
        self.smoke = smoke
        self.n_tokens = 0

        torch.manual_seed(0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        """Greedy-decode ``prompt``; returns the text with special tokens off."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        # output ids are [decoder_start_token, *generated, eos]; the decoder
        # start token comes from the model, not the prompt, so it is not
        # counted as generated throughput.
        self.n_tokens += output.shape[1] - 1
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def get_generator(smoke: bool = False) -> Generator:
    """Return the cached ``Generator`` for the pinned (or smoke) model id."""
    model_id = config.GEN_SMOKE if smoke else config.GEN_MODEL
    with _LOCK:
        generator = _GENERATORS.get(model_id)
        if generator is None:
            generator = _GENERATORS[model_id] = Generator(model_id, smoke=smoke)
    return generator
