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

    def train(self, texts: List[str]) -> None:
        """Learn BPE merges from a list of texts.

        Args:
            texts: list of training strings
        """
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


__all__ = [
    'SpecialTokens',
    'CharacterTokenizer',
    'BytePairEncoding',
    'WordPieceTokenizer',
    'SentencePieceTokenizer',
]
