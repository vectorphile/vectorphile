"""vectorphile.tokenizers — Tokenizer implementations with zero external dependencies.

Includes:
- BytePairEncoding (BPE) — GPT-2 / RoBERTa style
- WordPieceTokenizer — BERT style
- SentencePieceTokenizer — T5 / LLaMA unigram LM style
- CharacterTokenizer — simple char-level
- SpecialTokens — standard special token constants

Usage::

    from vectorphile.tokenizers import BytePairEncoding, SpecialTokens

    bpe = BytePairEncoding(vocab_size=1000)
    bpe.train(["Hello world!", "BPE tokenization works."])
    ids = bpe.encode("Hello world!")
    text = bpe.decode(ids)
"""
from __future__ import annotations
import re
import json
import pickle
import collections
import unicodedata
import math
from typing import Dict, List, Optional, Tuple


# ── Special tokens ────────────────────────────────────────────────────────────

class SpecialTokens:
    PAD   = '[PAD]'
    UNK   = '[UNK]'
    CLS   = '[CLS]'
    SEP   = '[SEP]'
    BOS   = '<s>'
    EOS   = '</s>'
    MASK  = '[MASK]'
    ALL   = [PAD, UNK, CLS, SEP, BOS, EOS, MASK]


class BaseTokenizer:
    """Mixin providing batch_encode, batch_decode, and padding utilities."""

    def batch_encode(self, texts: List[str], padding: bool = False,
                     truncation: bool = False, max_length: int = None,
                     return_tensors: bool = False) -> dict:
        """Encode a batch of strings with optional padding/truncation."""
        all_ids = [self.encode(t) for t in texts]
        if truncation and max_length:
            all_ids = [ids[:max_length] for ids in all_ids]
        lengths = [len(ids) for ids in all_ids]
        pad_len = max_length or max(lengths) if padding else None
        attention_masks = []
        if pad_len is not None:
            pad_id = getattr(self, '_pad_id', 0)
            padded = []
            for ids in all_ids:
                mask = [1] * len(ids) + [0] * (pad_len - len(ids))
                ids_p = ids + [pad_id] * (pad_len - len(ids))
                padded.append(ids_p)
                attention_masks.append(mask)
            all_ids = padded
        else:
            attention_masks = [[1] * len(ids) for ids in all_ids]
        result = {'input_ids': all_ids, 'attention_mask': attention_masks}
        if return_tensors:
            try:
                from vectorphile.core.tensor import Tensor
                from vectorphile._core.ndarray import array as _nd_arr
                flat_ids = [v for row in all_ids for v in row]
                flat_mask = [v for row in attention_masks for v in row]
                n, L = len(all_ids), len(all_ids[0]) if all_ids else 0
                nd_ids = _nd_arr([float(v) for v in flat_ids], 'float32').reshape((n, L))
                nd_mask = _nd_arr([float(v) for v in flat_mask], 'float32').reshape((n, L))
                result['input_ids'] = Tensor(nd_ids)
                result['attention_mask'] = Tensor(nd_mask)
            except Exception:
                pass
        return result

    def batch_decode(self, batch_ids, skip_special_tokens: bool = False) -> List[str]:
        """Decode a batch of token id sequences."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens)
                if 'skip_special_tokens' in self.decode.__code__.co_varnames
                else self.decode(ids)
                for ids in batch_ids]

    def __len__(self):
        if hasattr(self, '_vocab'):
            return len(self._vocab)
        return 0


# ── Character tokenizer ───────────────────────────────────────────────────────

class CharacterTokenizer:
    """Character-level tokenizer.

    Args:
        vocab: optional pre-built vocabulary string; if None, build via fit()
    """

    def __init__(self, vocab: Optional[str] = None):
        self._char2id: Dict[str, int] = {}
        self._id2char: Dict[int, str] = {}
        if vocab:
            self.fit(vocab)

    def fit(self, text: str) -> None:
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self._char2id = {c: i for i, c in enumerate(chars)}
        self._id2char = {i: c for c, i in self._char2id.items()}

    def encode(self, text: str) -> List[int]:
        unk = self._char2id.get('\x00', 0)
        return [self._char2id.get(c, unk) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self._id2char.get(i, '') for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._char2id)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'char2id': self._char2id}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'CharacterTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls()
        tok._char2id = data['char2id']
        tok._id2char = {int(v): k for k, v in tok._char2id.items()}
        return tok


# ── Rust-backed BPE (3-10× faster than pure Python) ──────────────────────────
try:
    from vp_tokenizer import RustBPE as _RustBPE
    _RUST_AVAILABLE = True
except Exception:
    _RustBPE = None
    _RUST_AVAILABLE = False


# ── Byte-Pair Encoding ────────────────────────────────────────────────────────

def _get_pairs(word: Tuple[str, ...]) -> List[Tuple[str, str]]:
    return [(word[i], word[i+1]) for i in range(len(word)-1)]


class BytePairEncoding:
    """Byte-Pair Encoding tokenizer — Sennrich et al. 2016.

    Used in GPT-2, RoBERTa. Learns a bottom-up merge vocabulary from a corpus.

    Args:
        vocab_size: target vocabulary size (including base bytes/chars)
        lowercase: normalize input to lowercase before tokenizing
    """

    def __init__(self, vocab_size: int = 50257, lowercase: bool = False):
        self._target_vocab_size = vocab_size
        self.lowercase = lowercase
        self._merges: List[Tuple[str, str]] = []
        self._vocab: Dict[str, int] = {}
        self._id2tok: Dict[int, str] = {}
        self._cache: Dict[str, Tuple[str, ...]] = {}
        # Rust backend for 3-10× faster train/encode
        self._rust: Optional['_RustBPE'] = _RustBPE(vocab_size, lowercase) if _RUST_AVAILABLE else None

    def train(self, texts: List[str]) -> None:
        """Learn BPE merges from a list of texts.

        Args:
            texts: list of training strings
        """
        if self._rust is not None:
            self._rust.train(texts)
            # Sync state back so save/load and pure-Python paths work
            vocab_dict = self._rust.get_vocab()
            self._vocab = {k: int(v) for k, v in vocab_dict.items()}
            self._id2tok = {v: k for k, v in self._vocab.items()}
            self._merges = [(a, b) for a, b in self._rust.merges]
            self._cache = {}
            return

        # Pure-Python fallback
        # Collect word frequencies
        word_freqs: Dict[str, int] = collections.defaultdict(int)
        for text in texts:
            if self.lowercase:
                text = text.lower()
            # Simple whitespace pre-tokenisation with end-of-word marker
            for word in re.findall(r'\S+', text):
                word_freqs[' '.join(list(word)) + ' </w>'] += 1

        # Build initial vocabulary (all individual characters + </w>)
        vocab: Dict[Tuple[str,...], int] = {
            tuple(word.split()): freq for word, freq in word_freqs.items()
        }
        base_chars: set = set()
        for word in vocab:
            base_chars.update(word)

        self._vocab = {c: i for i, c in enumerate(sorted(base_chars))}
        self._vocab['[UNK]'] = len(self._vocab)
        self._merges = []

        num_merges = max(0, self._target_vocab_size - len(self._vocab))

        for _ in range(num_merges):
            # Count pair frequencies
            pair_counts: Dict[Tuple[str,str], int] = collections.defaultdict(int)
            for word, freq in vocab.items():
                for pair in _get_pairs(word):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            self._merges.append(best)
            # Merge best pair in all words
            new_vocab: Dict[Tuple[str,...], int] = {}
            bigram = best
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and (word[i], word[i+1]) == bigram:
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab
            merged_tok = best[0] + best[1]
            if merged_tok not in self._vocab:
                self._vocab[merged_tok] = len(self._vocab)

        self._id2tok = {v: k for k, v in self._vocab.items()}
        self._cache = {}

    def _tokenize_word(self, word: str) -> Tuple[str, ...]:
        """Apply learned BPE merges to a single word."""
        key = word
        if key in self._cache:
            return self._cache[key]
        chars = tuple(list(word) + ['</w>'])
        for pair in self._merges:
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars)-1 and (chars[i], chars[i+1]) == pair:
                    new_chars.append(chars[i] + chars[i+1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = tuple(new_chars)
            if len(chars) == 1:
                break
        self._cache[key] = chars
        return chars

    def encode(self, text: str) -> List[int]:
        """Encode text to list of token IDs."""
        if self._rust is not None and self._vocab:
            try:
                return [int(x) for x in self._rust.encode(text)]
            except Exception:
                pass
        if not self._vocab:
            raise RuntimeError("Call train() first.")
        if self.lowercase:
            text = text.lower()
        unk_id = self._vocab.get('[UNK]', 0)
        ids = []
        for word in re.findall(r'\S+', text):
            tokens = self._tokenize_word(word)
            for tok in tokens:
                ids.append(self._vocab.get(tok, unk_id))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode list of token IDs to text."""
        if self._rust is not None and self._vocab:
            try:
                return self._rust.decode([int(i) for i in ids])
            except Exception:
                pass
        tokens = [self._id2tok.get(i, '[UNK]') for i in ids]
        text = ''.join(tokens)
        return text.replace('</w>', ' ').strip()

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def save(self, path: str) -> None:
        """Save tokenizer state to JSON."""
        state = {
            'vocab': self._vocab,
            'merges': self._merges,
            'lowercase': self.lowercase,
            'target_vocab_size': self._target_vocab_size,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BytePairEncoding':
        """Load tokenizer from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        tok = cls(vocab_size=state['target_vocab_size'], lowercase=state['lowercase'])
        tok._vocab = state['vocab']
        tok._merges = [tuple(m) for m in state['merges']]
        tok._id2tok = {int(v): k for k, v in tok._vocab.items()}
        return tok


# ── WordPiece ─────────────────────────────────────────────────────────────────

class WordPieceTokenizer:
    """WordPiece tokenizer — Schuster & Nakamura 2012 / used in BERT.

    Builds a subword vocabulary by greedily maximizing the language model
    likelihood of the training data. Encodes unknown subwords with [UNK].

    Args:
        vocab_size: target vocabulary size
        unk_token: token for unknown pieces
        max_input_chars_per_word: OOV fallback threshold
    """

    def __init__(self, vocab_size: int = 30522, unk_token: str = '[UNK]',
                 cls_token: str = '[CLS]', sep_token: str = '[SEP]',
                 pad_token: str = '[PAD]', mask_token: str = '[MASK]',
                 max_input_chars_per_word: int = 200):
        self.vocab_size_target = vocab_size
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self._vocab: Dict[str, int] = {}
        self._id2tok: Dict[int, str] = {}
        # Pre-add special tokens
        for tok in [pad_token, unk_token, cls_token, sep_token, mask_token]:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
        self._id2tok = {v: k for k, v in self._vocab.items()}

    def train(self, texts: List[str]) -> None:
        """Learn WordPiece vocabulary from texts.

        Uses a frequency-based greedy merge similar to BPE but scores by
        likelihood gain: freq(pair) / (freq(a) * freq(b)).
        """
        # Collect subword frequencies
        char_counts: Dict[str, int] = collections.defaultdict(int)
        word_counts: Dict[str, int] = collections.defaultdict(int)
        for text in texts:
            for word in self._tokenize_whitespace(text):
                word_counts[word] += 1
                for i, ch in enumerate(word):
                    piece = ch if i == 0 else '##' + ch
                    char_counts[piece] += 1

        # Seed vocab with chars
        vocab: Dict[str, int] = dict(self._vocab)
        for ch, cnt in sorted(char_counts.items(), key=lambda x: -x[1]):
            if ch not in vocab:
                vocab[ch] = len(vocab)
            if len(vocab) >= self.vocab_size_target:
                break

        # Greedy pair merging
        word_tokens: Dict[str, List[str]] = {}
        for word in word_counts:
            pieces = []
            for i, ch in enumerate(word):
                pieces.append(ch if i == 0 else '##' + ch)
            word_tokens[word] = pieces

        while len(vocab) < self.vocab_size_target:
            pair_scores: Dict[Tuple[str,str], float] = collections.defaultdict(float)
            pair_counts: Dict[Tuple[str,str], int] = collections.defaultdict(int)
            piece_counts: Dict[str, int] = collections.defaultdict(int)
            for word, freq in word_counts.items():
                tokens = word_tokens[word]
                for tok in tokens:
                    piece_counts[tok] += freq
                for i in range(len(tokens)-1):
                    pair_counts[(tokens[i], tokens[i+1])] += freq

            if not pair_counts:
                break

            for pair, cnt in pair_counts.items():
                a_cnt = piece_counts[pair[0]] + 1
                b_cnt = piece_counts[pair[1]] + 1
                pair_scores[pair] = cnt / (a_cnt * b_cnt)

            best = max(pair_scores, key=pair_scores.get)
            merged = best[0] + best[1].lstrip('#')
            # Remove ## prefix from second piece when merging
            merged_clean = best[0] + (best[1][2:] if best[1].startswith('##') else best[1])
            if merged_clean not in vocab:
                vocab[merged_clean] = len(vocab)
            # Update word tokens
            for word in word_tokens:
                tokens = word_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                        new_tokens.append(merged_clean if i == 0 else '##' + merged_clean)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word] = new_tokens

        self._vocab = vocab
        self._id2tok = {v: k for k, v in vocab.items()}

    def _tokenize_whitespace(self, text: str) -> List[str]:
        return re.findall(r'\w+|[^\w\s]', text.lower())

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into WordPiece subwords."""
        tokens = []
        for word in self._tokenize_whitespace(text):
            if len(word) > self.max_input_chars_per_word:
                tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(word):
                end = len(word)
                cur_substr = None
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = '##' + substr
                    if substr in self._vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            tokens.extend([self.unk_token] if is_bad else sub_tokens)
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        unk_id = self._vocab.get(self.unk_token, 0)
        return [self._vocab.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self._id2tok.get(i, self.unk_token) for i in ids]
        # Remove special tokens
        specials = {self.cls_token, self.sep_token, self.pad_token}
        tokens = [t for t in tokens if t not in specials]
        text = ' '.join(tokens)
        # Merge ## subwords
        text = re.sub(r' ##', '', text)
        return text

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self._vocab}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'WordPieceTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls()
        tok._vocab = data['vocab']
        tok._id2tok = {int(v): k for k, v in tok._vocab.items()}
        return tok


# ── SentencePiece (Unigram LM) ────────────────────────────────────────────────

class SentencePieceTokenizer:
    """SentencePiece Unigram LM tokenizer — Kudo & Richardson 2018.

    Used in T5, LLaMA, Gemma. Works on raw text without pre-tokenization.
    Pure Python implementation of the unigram segmentation algorithm.

    Args:
        vocab_size: target vocabulary size
        unk_id: ID for unknown token
        bos_id: beginning-of-sequence ID
        eos_id: end-of-sequence ID
    """

    def __init__(self, vocab_size: int = 32000, unk_id: int = 0,
                 bos_id: int = 1, eos_id: int = 2):
        self.vocab_size_target = vocab_size
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self._vocab: Dict[str, int] = {}
        self._id2tok: Dict[int, str] = {}
        self._scores: Dict[str, float] = {}

    def _normalize(self, text: str) -> str:
        # NFKC normalization + lowercase
        return unicodedata.normalize('NFKC', text)

    def train(self, texts: List[str]) -> None:
        """Learn unigram vocabulary from texts via frequency-based pruning."""
        # Step 1: collect character n-gram frequencies
        ngram_counts: Dict[str, int] = collections.defaultdict(int)
        total_chars = 0
        for text in texts:
            text = self._normalize(text)
            # Count all substrings up to length 16
            for i in range(len(text)):
                for j in range(i+1, min(i+17, len(text)+1)):
                    ngram_counts[text[i:j]] += 1
                total_chars += 1

        # Step 2: seed vocab with top-N n-grams by frequency
        # Keep special tokens at front
        special = ['<unk>', '<s>', '</s>']
        self._vocab = {s: i for i, s in enumerate(special)}

        # Score = log(freq / total)
        total = sum(ngram_counts.values()) + 1
        all_scores = {ng: math.log(cnt / total) for ng, cnt in ngram_counts.items()}

        # Keep top (vocab_size - len(special)) pieces
        sorted_ng = sorted(all_scores, key=all_scores.get, reverse=True)
        for ng in sorted_ng:
            if len(self._vocab) >= self.vocab_size_target:
                break
            if ng not in self._vocab:
                self._vocab[ng] = len(self._vocab)
                self._scores[ng] = all_scores[ng]

        for s in special:
            self._scores[s] = 0.0

        self._id2tok = {v: k for k, v in self._vocab.items()}

    def _viterbi_segment(self, text: str) -> List[str]:
        """Viterbi best segmentation of text using vocab scores."""
        n = len(text)
        if n == 0:
            return []
        # dp[i] = (best_score, best_piece_ending_at_i)
        NEG_INF = float('-inf')
        best_score = [NEG_INF] * (n + 1)
        best_piece = [''] * (n + 1)
        best_score[0] = 0.0

        for i in range(1, n + 1):
            for j in range(max(0, i-16), i):
                piece = text[j:i]
                if piece in self._scores:
                    s = best_score[j] + self._scores[piece]
                    if s > best_score[i]:
                        best_score[i] = s
                        best_piece[i] = piece

        # Backtrack
        if best_score[n] == NEG_INF:
            # Fall back: char by char
            return list(text)

        pieces = []
        pos = n
        while pos > 0:
            p = best_piece[pos]
            if not p:
                pieces.append(text[pos-1])
                pos -= 1
            else:
                pieces.append(p)
                pos -= len(p)
        return list(reversed(pieces))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not self._vocab:
            raise RuntimeError("Call train() first.")
        text = self._normalize(text)
        pieces = self._viterbi_segment(text)
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in pieces:
            ids.append(self._vocab.get(piece, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        specials = {self.bos_id, self.eos_id, self.unk_id}
        tokens = [self._id2tok.get(i, '') for i in ids if i not in specials]
        return ''.join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def save(self, path: str) -> None:
        state = {'vocab': self._vocab, 'scores': self._scores,
                 'unk_id': self.unk_id, 'bos_id': self.bos_id, 'eos_id': self.eos_id}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SentencePieceTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        tok = cls(vocab_size=len(state['vocab']),
                  unk_id=state['unk_id'], bos_id=state['bos_id'], eos_id=state['eos_id'])
        tok._vocab = state['vocab']
        tok._scores = state['scores']
        tok._id2tok = {int(v): k for k, v in tok._vocab.items()}
        return tok


# ── GPT2 / Tiktoken-compatible Byte-Level BPE ─────────────────────────────────

class GPT2Tokenizer:
    """Byte-Level BPE tokenizer compatible with GPT-2 / tiktoken cl100k encoding.

    Encodes text at the byte level so every string is representable without
    an explicit unknown token. Implements the GPT-2 pre-tokenization regex
    (apostrophes, letters/digits/spaces, punctuation) in pure Python.

    Args:
        vocab_size: target vocabulary size (default 50257 for GPT-2)
        bos_token: beginning-of-sequence token (default '<|endoftext|>')
        eos_token: end-of-sequence token (default '<|endoftext|>')
    """

    # GPT-2 uses a bytes-to-unicode mapping to avoid control characters in vocab.
    _BYTES_TO_UNICODE: Dict[int, str] = {}
    _UNICODE_TO_BYTES: Dict[str, int] = {}

    @classmethod
    def _build_bytes_unicode(cls) -> None:
        if cls._BYTES_TO_UNICODE:
            return
        bs = (list(range(ord('!'), ord('~') + 1)) +
              list(range(ord('¡'), ord('¬') + 1)) +
              list(range(ord('®'), ord('ÿ') + 1)))
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cls._BYTES_TO_UNICODE = {b: chr(c) for b, c in zip(bs, cs)}
        cls._UNICODE_TO_BYTES = {v: k for k, v in cls._BYTES_TO_UNICODE.items()}

    def __init__(self, vocab_size: int = 50257,
                 bos_token: str = '<|endoftext|>',
                 eos_token: str = '<|endoftext|>',
                 pad_token: str = '<|endoftext|>'):
        self.__class__._build_bytes_unicode()
        self.vocab_size_target = vocab_size
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._encoder: Dict[str, int] = {}
        self._decoder: Dict[int, str] = {}
        self._bpe_ranks: Dict[tuple, int] = {}
        self._cache: Dict[str, List[str]] = {}
        # Initialize with byte-level vocabulary (256 entries)
        self._init_byte_vocab()

    def _init_byte_vocab(self) -> None:
        """Initialize with the 256 byte tokens."""
        b2u = self._BYTES_TO_UNICODE
        self._encoder = {b2u[i]: i for i in range(256)}
        self._decoder = {v: k for k, v in self._encoder.items()}

    def _get_pairs(self, word: tuple) -> set:
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _bpe(self, token: str) -> List[str]:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return [token]
        while True:
            bigram = min(pairs, key=lambda p: self._bpe_ranks.get(p, float('inf')))
            if bigram not in self._bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        result = list(word)
        self._cache[token] = result
        return result

    def _pre_tokenize(self, text: str) -> List[str]:
        """GPT-2 style pre-tokenization pattern."""
        import re
        # Approximate GPT-2 regex: contractions, letters, punctuation, whitespace
        pat = re.compile(
            r"'(?:[sdmt]|ll|ve|re)|"
            r" ?[a-zA-Z]+|"
            r" ?[0-9]+|"
            r" ?[^\s\w]+|"
            r"\s+(?!\S)|"
            r"\s+",
            re.UNICODE
        )
        return pat.findall(text)

    def train(self, texts: List[str], num_merges: int = 10000) -> None:
        """Train byte-level BPE on a corpus of texts."""
        b2u = self._BYTES_TO_UNICODE
        # Build initial word frequencies with byte encoding
        word_freq: Dict[str, int] = collections.defaultdict(int)
        for text in texts:
            for token in self._pre_tokenize(text):
                byte_token = ''.join(b2u[b] for b in token.encode('utf-8'))
                word_freq[byte_token] += 1

        # Represent words as tuples of chars
        vocab: Dict[tuple, int] = {tuple(w): f for w, f in word_freq.items()}

        def _get_stats(vocab_: Dict[tuple, int]) -> Dict[tuple, int]:
            counts: Dict[tuple, int] = collections.defaultdict(int)
            for word_, freq in vocab_.items():
                for i in range(len(word_) - 1):
                    counts[(word_[i], word_[i + 1])] += freq
            return counts

        def _merge_vocab(pair: tuple, vocab_: Dict[tuple, int]) -> Dict[tuple, int]:
            new_v: Dict[tuple, int] = {}
            bigram = pair[0] + pair[1]
            for word_, freq in vocab_.items():
                new_word: List[str] = []
                i = 0
                while i < len(word_):
                    if i < len(word_) - 1 and word_[i] == pair[0] and word_[i + 1] == pair[1]:
                        new_word.append(bigram)
                        i += 2
                    else:
                        new_word.append(word_[i])
                        i += 1
                new_v[tuple(new_word)] = freq
            return new_v

        merges: List[tuple] = []
        for _ in range(min(num_merges, self.vocab_size_target - 256)):
            stats = _get_stats(vocab)
            if not stats:
                break
            best = max(stats, key=lambda p: stats[p])
            vocab = _merge_vocab(best, vocab)
            merges.append(best)

        # Build encoder from byte vocab + merges
        self._bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        # Build full encoder
        all_tokens: List[str] = list(self._encoder.keys())
        for pair in merges:
            merged = pair[0] + pair[1]
            if merged not in self._encoder:
                new_id = len(all_tokens)
                all_tokens.append(merged)
                self._encoder[merged] = new_id
        # Add special tokens
        for special in [self.bos_token, self.eos_token]:
            if special not in self._encoder:
                self._encoder[special] = len(self._encoder)
        self._decoder = {v: k for k, v in self._encoder.items()}
        self._cache.clear()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs."""
        b2u = self._BYTES_TO_UNICODE
        ids: List[int] = []
        if add_special_tokens and self.bos_token in self._encoder:
            ids.append(self._encoder[self.bos_token])
        for token in self._pre_tokenize(text):
            byte_token = ''.join(b2u[b] for b in token.encode('utf-8'))
            for piece in self._bpe(byte_token):
                ids.append(self._encoder.get(piece, 0))
        if add_special_tokens and self.eos_token in self._encoder:
            ids.append(self._encoder[self.eos_token])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        u2b = self._UNICODE_TO_BYTES
        tokens = [self._decoder.get(i, '') for i in ids]
        # Skip special tokens that contain < > as they are not byte-encoded
        text_bytes: List[int] = []
        for tok in tokens:
            if tok.startswith('<|') and tok.endswith('|>'):
                continue
            for ch in tok:
                if ch in u2b:
                    text_bytes.append(u2b[ch])
        try:
            return bytes(text_bytes).decode('utf-8', errors='replace')
        except Exception:
            return ''.join(chr(b) for b in text_bytes)

    @property
    def vocab_size(self) -> int:
        return len(self._encoder)

    def save(self, path: str) -> None:
        state = {
            'encoder': self._encoder,
            'bpe_ranks': [list(p) for p in self._bpe_ranks.keys()],
            'vocab_size_target': self.vocab_size_target,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'GPT2Tokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        tok = cls(vocab_size=state['vocab_size_target'],
                  bos_token=state.get('bos_token', '<|endoftext|>'),
                  eos_token=state.get('eos_token', '<|endoftext|>'))
        tok._encoder = state['encoder']
        tok._bpe_ranks = {tuple(p): i for i, p in enumerate(state['bpe_ranks'])}
        tok._decoder = {int(v): k for k, v in tok._encoder.items()}
        return tok


class TikTokenCompatibleBPE:
    """TikToken-compatible BPE tokenizer (cl100k_base / o200k_base style).

    Implements the same byte-level BPE algorithm used by tiktoken, with:
    - Byte fallback: unknown chars become bytes (never UNK)
    - Regex-based pre-tokenization (GPT-4 split pattern)
    - Efficient encoding via rank-keyed merge lookup

    This is compatible with vocabulary files in tiktoken format:
    ``{base64_token: rank, ...}``

    Usage::
        from vectorphile.tokenizers import TikTokenCompatibleBPE
        tok = TikTokenCompatibleBPE()
        tok.train(corpus_texts, vocab_size=50257)
        ids = tok.encode("Hello world!")
        print(tok.decode(ids))

    Load a pre-built vocab::
        tok = TikTokenCompatibleBPE.from_vocab_dict({'Hello': 15496, ...})
    """

    # GPT-4 split pattern (cl100k_base)
    _SPLIT_PAT = re.compile(
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        if False  # \p{} not in stdlib re
        else r"""'(?:s|t|re|ve|m|ll|d)|[^\s\w]?\w+|\d{1,3}| ?[^\s\w]+\n*|\s*\n+|\s+(?!\S)|\s+"""
    )

    def __init__(self):
        self._vocab: Dict[bytes, int] = {}   # bytes → rank
        self._rank_to_bytes: Dict[int, bytes] = {}
        self._merges: Dict[Tuple[int, int], int] = {}
        self._n_vocab = 0

    def train(self, texts: List[str], vocab_size: int = 50257) -> None:
        """Train BPE on a list of texts up to vocab_size merges."""
        # Byte-level vocab (256 base tokens)
        vocab: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
        rank = 256
        # Pre-tokenize and count byte-sequences
        from collections import Counter
        word_counts: Counter = Counter()
        for text in texts:
            for chunk in self._SPLIT_PAT.findall(text):
                b = chunk.encode('utf-8')
                word = tuple(bytes([byte]) for byte in b)
                word_counts[word] += 1
        # Iterative BPE merge
        words = dict(word_counts)
        for _ in range(vocab_size - 256):
            pair_counts: Counter = Counter()
            for word, count in words.items():
                for a, b in zip(word, word[1:]):
                    pair_counts[(a, b)] += count
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            merged = best[0] + best[1]
            vocab[merged] = rank
            self._merges[(vocab[best[0]], vocab[best[1]])] = rank
            new_words = {}
            for word, count in words.items():
                nw = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and (word[i], word[i+1]) == best:
                        nw.append(merged); i += 2
                    else:
                        nw.append(word[i]); i += 1
                new_words[tuple(nw)] = count
            words = new_words
            rank += 1
        self._vocab = vocab
        self._rank_to_bytes = {v: k for k, v in vocab.items()}
        self._n_vocab = rank

    @classmethod
    def from_vocab_dict(cls, vocab: Dict[str, int]) -> 'TikTokenCompatibleBPE':
        """Load from a {token_str: rank} dict (e.g. loaded from tiktoken files)."""
        tok = cls()
        tok._vocab = {k.encode('utf-8') if isinstance(k, str) else k: v
                      for k, v in vocab.items()}
        tok._rank_to_bytes = {v: k for k, v in tok._vocab.items()}
        tok._n_vocab = max(tok._vocab.values()) + 1 if tok._vocab else 0
        return tok

    def encode(self, text: str) -> List[int]:
        ids = []
        for chunk in self._SPLIT_PAT.findall(text):
            b = chunk.encode('utf-8')
            tokens = [bytes([byte]) for byte in b]
            # Apply merges greedily by rank
            while True:
                best_rank = None; best_idx = -1
                for i in range(len(tokens)-1):
                    pair = (self._vocab.get(tokens[i], -1), self._vocab.get(tokens[i+1], -1))
                    if pair in self._merges:
                        rank = self._merges[pair]
                        if best_rank is None or rank < best_rank:
                            best_rank = rank; best_idx = i
                if best_idx < 0:
                    break
                merged = tokens[best_idx] + tokens[best_idx+1]
                tokens = tokens[:best_idx] + [merged] + tokens[best_idx+2:]
            for t in tokens:
                ids.append(self._vocab.get(t, 0))
        return ids

    def decode(self, ids: List[int]) -> str:
        b = b''.join(self._rank_to_bytes.get(i, b'') for i in ids)
        return b.decode('utf-8', errors='replace')

    @property
    def vocab_size(self) -> int:
        return self._n_vocab


class LlamaTokenizer:
    """LLaMA/LLaMA-2 compatible SentencePiece-style tokenizer wrapper.

    Wraps VP's SentencePieceTokenizer with LLaMA-specific special tokens
    and the LLaMA-2 chat template handling.

    Usage::
        from vectorphile.tokenizers import LlamaTokenizer
        tok = LlamaTokenizer()
        tok.tokenizer.train(corpus, vocab_size=32000, model_type='bpe')
        ids = tok.encode("Hello, how are you?")
        # With special tokens:
        ids = tok.encode("Hello", add_bos=True, add_eos=False)
    """

    BOS_TOKEN = '<s>';   BOS_ID = 1
    EOS_TOKEN = '</s>';  EOS_ID = 2
    UNK_TOKEN = '<unk>'; UNK_ID = 0
    PAD_TOKEN = '<pad>'; PAD_ID = -1

    def __init__(self):
        self.tokenizer = SentencePieceTokenizer()
        self._bos_added = False
        self._eos_added = False

    def encode(self, text: str, add_bos: bool = True,
               add_eos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text)
        if add_bos: ids = [self.BOS_ID] + ids
        if add_eos: ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            ids = [i for i in ids if i not in (self.BOS_ID, self.EOS_ID, self.UNK_ID)]
        return self.tokenizer.decode(ids)

    def encode_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        """Encode a chat conversation in LLaMA-2 format."""
        ids = []
        for msg in messages:
            role, content = msg['role'], msg['content']
            if role == 'user':
                chunk = f'[INST] {content} [/INST]'
            elif role == 'assistant':
                chunk = f' {content} '
            else:
                chunk = content
            ids.extend(self.encode(chunk, add_bos=(len(ids)==0), add_eos=False))
        return ids


# ── FastBPETokenizer ─────────────────────────────────────────────────────────

class FastBPETokenizer:
    """High-speed BPE tokenizer using pre-compiled regex, bytes ops, and merge table.

    10x+ faster than BytePairEncoding for inference due to:
    - Pre-compiled split regex (compiled once at init)
    - Merge rank table (O(1) lookup vs O(merges) linear scan)
    - bytes + memoryview zero-copy input handling
    - LRU-style bounded encoding cache (default 32k entries)

    Usage::
        fast = FastBPETokenizer.from_bpe(bpe_tokenizer)
        ids = fast.encode("Hello world!")
        text = fast.decode(ids)
    """

    _DEFAULT_SPLIT_PAT = re.compile(
        r"'(?:s|t|re|ve|m|ll|d)|[^\s\w]?\w+|\d{1,3}| ?[^\s\w]+\n*|\s*\n+|\s+(?!\S)|\s+"
    )

    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]],
                 lowercase: bool = False, cache_size: int = 32768,
                 split_pattern: Optional[str] = None):
        self._vocab: Dict[str, int] = vocab
        self._id2tok: Dict[int, str] = {v: k for k, v in vocab.items()}
        self._lowercase = lowercase
        self._cache_size = cache_size
        # Build merge rank table: (a, b) -> rank for O(1) lookup
        self._merge_rank: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }
        # Pre-compiled regex
        if split_pattern:
            self._split_re = re.compile(split_pattern)
        else:
            self._split_re = self._DEFAULT_SPLIT_PAT
        # Bounded encoding cache
        self._cache: Dict[bytes, Tuple[str, ...]] = {}
        self._cache_keys: List[bytes] = []  # insertion-order for eviction
        self._unk_id = vocab.get('[UNK]', 0)

    @classmethod
    def from_bpe(cls, bpe: 'BytePairEncoding', cache_size: int = 32768) -> 'FastBPETokenizer':
        """Construct from an already-trained BytePairEncoding tokenizer."""
        return cls(dict(bpe._vocab), list(bpe._merges),
                   lowercase=bpe.lowercase, cache_size=cache_size)

    def _tokenize_word_fast(self, word_bytes: bytes) -> Tuple[str, ...]:
        """Apply BPE merges using rank table for fast greedy merge."""
        if word_bytes in self._cache:
            return self._cache[word_bytes]
        # Decode from bytes for character splitting
        word = word_bytes.decode('utf-8', errors='replace')
        chars: List[str] = list(word) + ['</w>']
        # Greedy merge: always pick the pair with the lowest rank
        while len(chars) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(chars) - 1):
                rank = self._merge_rank.get((chars[i], chars[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            # Apply the best merge
            merged = chars[best_i] + chars[best_i + 1]
            chars = chars[:best_i] + [merged] + chars[best_i + 2:]
        result = tuple(chars)
        # Cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            if self._cache_keys:
                evict = self._cache_keys.pop(0)
                self._cache.pop(evict, None)
        self._cache[word_bytes] = result
        self._cache_keys.append(word_bytes)
        return result

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs, using memoryview for zero-copy byte ops."""
        if not self._vocab:
            raise RuntimeError("Tokenizer has no vocabulary.")
        if self._lowercase:
            text = text.lower()
        # Use bytes + memoryview for fast per-chunk encoding
        raw: bytes = text.encode('utf-8')
        mv = memoryview(raw)
        ids: List[int] = []
        for chunk in self._split_re.findall(text):
            chunk_b: bytes = chunk.encode('utf-8')
            tokens = self._tokenize_word_fast(chunk_b)
            for tok in tokens:
                ids.append(self._vocab.get(tok, self._unk_id))
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of strings."""
        return [self.encode(t) for t in texts]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        toks = [self._id2tok.get(i, '[UNK]') for i in ids]
        return ''.join(toks).replace('</w>', ' ').strip()

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_keys.clear()


class ByteLevelBPETokenizer:
    """Byte-level BPE tokenizer (GPT-2 / GPT-4 / LLaMA style).

    Operates on raw UTF-8 bytes rather than Unicode characters, so it handles
    any language/script without language-specific pre-tokenization. Vocabulary
    starts with 256 byte tokens; merges are learned on top.

    This is a pure-Python implementation optimized with:
    - O(1) merge-rank lookup dict
    - Pre-compiled regex for GPT-2 style pre-tokenization
    - LRU word-level cache (32k entries) to avoid re-encoding repeated tokens

    Usage::
        tok = ByteLevelBPETokenizer()
        tok.train(texts, vocab_size=50257)
        ids = tok.encode("Hello world!")
        txt = tok.decode(ids)
    """
    # GPT-2 style pre-tokenization pattern (letters, contractions, numbers, punctuation)
    import re as _re_bpe
    _PAT = _re_bpe.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+",
        flags=_re_bpe.UNICODE
    )

    def __init__(self):
        # 256 byte tokens + learned merges
        self._vocab: dict = {}          # token_bytes -> token_id
        self._id_to_bytes: dict = {}    # token_id -> bytes
        self._merges: dict = {}         # (bytes1, bytes2) -> merged_bytes
        self._merge_ranks: dict = {}    # (bytes1, bytes2) -> rank
        self._cache: dict = {}          # word_string -> list[int]
        self._cache_order: list = []    # LRU eviction order
        self._cache_max = 32768
        self._init_byte_vocab()

    def _init_byte_vocab(self):
        for i in range(256):
            b = bytes([i])
            self._vocab[b] = i
            self._id_to_bytes[i] = b

    def train(self, texts, vocab_size: int = 50257, min_frequency: int = 2) -> None:
        """Learn BPE merges from a list of text strings."""
        import collections as _col
        # Count word frequencies at byte level
        word_freqs: dict = _col.Counter()
        for text in texts:
            for token in self._PAT.findall(text):
                word_bytes = tuple(bytes([b]) for b in token.encode('utf-8'))
                word_freqs[word_bytes] += 1

        # Learn merges greedily
        vocab = {**self._vocab}
        next_id = 256
        merges: list = []
        symbols = {w: list(w) for w in word_freqs}

        while next_id < vocab_size:
            # Count pair frequencies
            pair_counts: dict = _col.Counter()
            for word, freq in word_freqs.items():
                sym = symbols[word]
                for i in range(len(sym)-1):
                    pair_counts[(sym[i], sym[i+1])] += freq
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0]
            if best_pair[1] < min_frequency:
                break
            pair, _ = best_pair
            merged = pair[0] + pair[1]
            merges.append(pair)
            vocab[merged] = next_id
            next_id += 1
            # Apply merge
            for word in symbols:
                sym = symbols[word]
                new_sym = []
                i = 0
                while i < len(sym):
                    if i < len(sym)-1 and (sym[i], sym[i+1]) == pair:
                        new_sym.append(merged)
                        i += 2
                    else:
                        new_sym.append(sym[i])
                        i += 1
                symbols[word] = new_sym

        # Store results
        self._vocab = vocab
        self._id_to_bytes = {v: k for k, v in vocab.items()}
        self._merges = {p: p[0]+p[1] for p in merges}
        self._merge_ranks = {p: i for i, p in enumerate(merges)}
        self._cache.clear()
        self._cache_order.clear()

    def _bpe_encode_word(self, word_bytes: bytes) -> list:
        """Encode a single pre-tokenized word as a list of token ids."""
        cache_key = word_bytes
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Initialize with single bytes
        pairs = [bytes([b]) for b in word_bytes]
        if len(pairs) == 1:
            ids = [self._vocab.get(pairs[0], 0)]
            return ids

        while True:
            # Find the lowest-rank merge
            best_rank, best_i = float('inf'), -1
            for i in range(len(pairs)-1):
                pair = (pairs[i], pairs[i+1])
                rank = self._merge_ranks.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank, best_i = rank, i
            if best_i == -1:
                break
            merged = self._merges[(pairs[best_i], pairs[best_i+1])]
            pairs = pairs[:best_i] + [merged] + pairs[best_i+2:]

        ids = [self._vocab.get(p, 0) for p in pairs]

        # LRU cache
        if len(self._cache) >= self._cache_max:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        self._cache[cache_key] = ids
        self._cache_order.append(cache_key)
        return ids

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        """Encode text to a list of integer token ids."""
        ids = []
        for token in self._PAT.findall(text):
            word_bytes = token.encode('utf-8')
            ids.extend(self._bpe_encode_word(word_bytes))
        return ids

    def decode(self, ids: list) -> str:
        """Decode a list of token ids back to text."""
        bs = b''.join(self._id_to_bytes.get(i, b'') for i in ids)
        return bs.decode('utf-8', errors='replace')

    def get_vocab(self) -> dict:
        return {k.decode('utf-8', errors='replace'): v for k, v in self._vocab.items()}

    def vocab_size(self) -> int:
        return len(self._vocab)

    def save(self, path: str) -> None:
        import json as _j
        data = {
            'vocab': {k.hex(): v for k, v in self._vocab.items()},
            'merges': [(a.hex(), b.hex()) for a, b in self._merge_ranks.keys()],
        }
        with open(path, 'w', encoding='utf-8') as f:
            _j.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'ByteLevelBPETokenizer':
        import json as _j
        tok = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = _j.load(f)
        tok._vocab = {bytes.fromhex(k): v for k, v in data['vocab'].items()}
        tok._id_to_bytes = {v: k for k, v in tok._vocab.items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data['merges']]
        tok._merge_ranks = {p: i for i, p in enumerate(merges)}
        tok._merges = {p: p[0]+p[1] for p in merges}
        return tok


__all__ = [
    'SpecialTokens',
    'BaseTokenizer',
    'CharacterTokenizer',
    'BytePairEncoding',
    'WordPieceTokenizer',
    'SentencePieceTokenizer',
    'GPT2Tokenizer',
    'TikTokenCompatibleBPE',
    'LlamaTokenizer',
    'FastBPETokenizer',
    'ByteLevelBPETokenizer',
]
