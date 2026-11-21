"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        self.merges = {}
        ids = list(text.encode('utf-8'))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=lambda p: stats[p])
            ids = merge(ids, pair, i + 256)
            self.merges[pair] = i + 256
            if verbose:
                print(f"Iteration: {i+1} | Selected Pair: {pair}, frequency: {stats[pair]} | IDX: {i+256}")
        self.vocab = self._build_vocab()

    def decode(self, ids):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError
