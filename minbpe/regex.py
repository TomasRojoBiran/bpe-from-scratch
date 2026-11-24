"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base import Tokenizer, get_stats, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.regex = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        self.merges = {}
        text = self.regex.findall(text)
        ids = [list(chunk.encode('utf-8')) for chunk in text]
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                stats = get_stats(chunk, counts=stats)
            if not stats:
                break
            pair = max(stats, key=lambda p: stats[p])
            ids = [merge(chunk, pair, i + 256) for chunk in ids]
            self.merges[pair] = i + 256
            if verbose:
                print(f"Iteration: {i+1} | Selected Pair: {pair}, frequency: {stats[pair]} | IDX: {i+256}")
        self.vocab = self._build_vocab()

    def register_special_tokens(self, special_tokens):
        raise NotImplementedError

    def decode(self, ids):
        # Reuse the existing decoding logic and then in step 4 adapt it for special tokens
        raise NotImplementedError

    def _encode_chunk(self, text_bytes):
        raise NotImplementedError

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        raise NotImplementedError

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # Reuse the existing decoding logic and then in step 4 adapt it for special tokens
        raise NotImplementedError
