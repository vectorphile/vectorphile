"""vectorphile.data.datasets — Standard datasets + Dataset/DataLoader utilities."""
from __future__ import annotations
import os
import gzip
import struct
import pickle
import hashlib
import tarfile
import urllib.request
import urllib.error
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# ── Dataset base ──────────────────────────────────────────────────────────────

class Dataset:
    """Abstract base class for datasets (mirrors torch.utils.data.Dataset)."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = [np.asarray(t.data if hasattr(t, 'data') else t) for t in tensors]

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self._cumulative = []
        c = 0
        for d in datasets:
            c += len(d)
            self._cumulative.append(c)

    def __len__(self):
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, idx):
        for i, cum in enumerate(self._cumulative):
            if idx < cum:
                offset = idx - (self._cumulative[i-1] if i > 0 else 0)
                return self.datasets[i][offset]
        raise IndexError(idx)


# ── Samplers ──────────────────────────────────────────────────────────────────

class Sampler:
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, dataset: Dataset):
        self.n = len(dataset)

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


class RandomSampler(Sampler):
    def __init__(self, dataset: Dataset, replacement: bool = False, num_samples: Optional[int] = None):
        self.n = len(dataset)
        self.replacement = replacement
        self._num_samples = num_samples or self.n

    def __iter__(self):
        if self.replacement:
            return iter(np.random.randint(0, self.n, self._num_samples).tolist())
        return iter(np.random.permutation(self.n)[:self._num_samples].tolist())

    def __len__(self):
        return self._num_samples


class WeightedRandomSampler(Sampler):
    def __init__(self, weights: Sequence[float], num_samples: int, replacement: bool = True):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        indices = np.random.choice(len(self.weights), self.num_samples,
                                   replace=self.replacement, p=self.weights)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class DistributedSampler(Sampler):
    """Subset sampler for distributed training."""

    def __init__(self, dataset: Dataset, num_replicas: int = 1, rank: int = 0,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        self.n = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = rng.permutation(self.n).tolist()
        else:
            indices = list(range(self.n))
        if not self.drop_last:
            pad = self.num_replicas - len(indices) % self.num_replicas
            if pad != self.num_replicas:
                indices += indices[:pad]
        else:
            indices = indices[:len(indices) - len(indices) % self.num_replicas]
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        return self.n // self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# ── DataLoader ────────────────────────────────────────────────────────────────

def _default_collate(batch):
    """Stack a list of samples into a batched array."""
    if isinstance(batch[0], (tuple, list)):
        return type(batch[0])(_default_collate(list(s[i] for s in batch))
                               for i in range(len(batch[0])))
    if isinstance(batch[0], dict):
        return {k: _default_collate([s[k] for s in batch]) for k in batch[0]}
    try:
        arr = np.stack([np.asarray(s) for s in batch], axis=0)
        return arr
    except Exception:
        return batch


class DataLoader:
    """Iterable DataLoader (mirrors torch.utils.data.DataLoader).

    Supports shuffle, custom samplers, collate_fn, drop_last, and num_workers
    (num_workers > 0 uses concurrent.futures ThreadPoolExecutor as a lightweight
    parallel prefetch — not true multiprocessing).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler=None,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.collate_fn = collate_fn or _default_collate
        self.batch_sampler = batch_sampler

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

    def __len__(self) -> int:
        n = len(self.sampler) if hasattr(self.sampler, '__len__') else len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(iter(self.sampler))
        # Form batches
        batches = []
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            batches.append(batch_idx)

        if self.num_workers > 0:
            yield from self._iter_parallel(batches)
        else:
            for batch_idx in batches:
                samples = [self.dataset[i] for i in batch_idx]
                yield self.collate_fn(samples)

    def _iter_parallel(self, batches):
        from concurrent.futures import ThreadPoolExecutor

        def fetch(idx_list):
            return self.collate_fn([self.dataset[i] for i in idx_list])

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = [ex.submit(fetch, b) for b in batches]
            for f in futures:
                yield f.result()


# ── Download helper ───────────────────────────────────────────────────────────

def _download(url: str, dest_path: str, show_progress: bool = True):
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    if os.path.exists(dest_path):
        return
    print(f'Downloading {url}')
    try:
        def reporthook(count, block_size, total_size):
            if show_progress and total_size > 0:
                pct = count * block_size * 100 / total_size
                print(f'\r  {min(pct, 100):.1f}%', end='', flush=True)
        urllib.request.urlretrieve(url, dest_path, reporthook)
        if show_progress:
            print()
    except Exception as e:
        raise RuntimeError(f'Download failed: {e}')


# ── MNIST ─────────────────────────────────────────────────────────────────────

class MNIST(Dataset):
    """MNIST handwritten digits dataset.

    Downloads from Yann LeCun's mirror (or local cache) on first use.
    """

    _URLS = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images':  'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels':  'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    }

    def __init__(
        self,
        root: str = '~/.vectorphile/datasets/mnist',
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()
        self.data, self.targets = self._load()

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        for name, url in self._URLS.items():
            fname = os.path.join(self.root, url.split('/')[-1])
            _download(url, fname)

    def _load(self):
        prefix = 'train' if self.train else 'test'
        # Find files
        img_gz = [f for f in os.listdir(self.root)
                  if f.startswith(prefix + '-images') and f.endswith('.gz')]
        lbl_gz = [f for f in os.listdir(self.root)
                  if f.startswith(prefix + '-labels') and f.endswith('.gz')]
        if not img_gz or not lbl_gz:
            raise FileNotFoundError(
                f"MNIST files not found in {self.root}. Set download=True."
            )
        with gzip.open(os.path.join(self.root, img_gz[0]), 'rb') as f:
            magic, n, h, w = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w)
        with gzip.open(os.path.join(self.root, lbl_gz[0]), 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        lbl = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lbl = self.target_transform(lbl)
        return img, lbl

    @property
    def classes(self):
        return [str(i) for i in range(10)]


# ── CIFAR-10 ──────────────────────────────────────────────────────────────────

class CIFAR10(Dataset):
    """CIFAR-10 dataset (32×32 color images, 10 classes)."""

    _URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    _TRAIN_FILES = [f'data_batch_{i}' for i in range(1, 6)]
    _TEST_FILE = 'test_batch'
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(
        self,
        root: str = '~/.vectorphile/datasets/cifar10',
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()
        self.data, self.targets = self._load()

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        fname = os.path.join(self.root, 'cifar-10-python.tar.gz')
        _download(self._URL, fname)
        if not os.path.exists(os.path.join(self.root, 'cifar-10-batches-py')):
            with tarfile.open(fname, 'r:gz') as tar:
                tar.extractall(self.root)

    def _load_batch(self, path: str):
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        data = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC
        labels = np.array(d[b'labels'])
        return data, labels

    def _load(self):
        base = os.path.join(self.root, 'cifar-10-batches-py')
        if not os.path.exists(base):
            raise FileNotFoundError(f"CIFAR-10 not found in {self.root}. Set download=True.")
        if self.train:
            arrays = [self._load_batch(os.path.join(base, f)) for f in self._TRAIN_FILES]
            data = np.concatenate([a[0] for a in arrays], axis=0)
            targets = np.concatenate([a[1] for a in arrays], axis=0)
        else:
            data, targets = self._load_batch(os.path.join(base, self._TEST_FILE))
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        lbl = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lbl = self.target_transform(lbl)
        return img, lbl


# ── TinyShakespeare ───────────────────────────────────────────────────────────

class TinyShakespeare(Dataset):
    """Character-level Shakespeare dataset for LM training."""

    _URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    def __init__(
        self,
        root: str = '~/.vectorphile/datasets/tinyshakespeare',
        split: str = 'train',
        block_size: int = 256,
        download: bool = True,
    ):
        self.root = os.path.expanduser(root)
        self.block_size = block_size
        os.makedirs(self.root, exist_ok=True)
        self._path = os.path.join(self.root, 'input.txt')
        if download:
            _download(self._URL, self._path)
        with open(self._path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        n = int(0.9 * len(data))
        self.data = data[:n] if split == 'train' else data[n:]

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi.get(c, 0) for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return ''.join(self.itos.get(int(i), '?') for i in ids)


# ── IMDb ──────────────────────────────────────────────────────────────────────

class IMDb(Dataset):
    """IMDb movie review sentiment dataset.

    Downloads the aclImdb dataset from Stanford NLP on first use.
    Returns (text: str, label: int) where label 0=neg, 1=pos.
    """

    _URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    def __init__(
        self,
        root: str = '~/.vectorphile/datasets/imdb',
        split: str = 'train',
        download: bool = True,
    ):
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)
        if download:
            fname = os.path.join(self.root, 'aclImdb_v1.tar.gz')
            _download(self._URL, fname)
            extracted = os.path.join(self.root, 'aclImdb')
            if not os.path.exists(extracted):
                with tarfile.open(fname, 'r:gz') as tar:
                    tar.extractall(self.root)
        self.samples = self._load(split)

    def _load(self, split: str) -> List[Tuple[str, int]]:
        base = os.path.join(self.root, 'aclImdb', split)
        if not os.path.exists(base):
            raise FileNotFoundError(f"IMDb not found at {base}. Set download=True.")
        samples = []
        for label_idx, label_name in enumerate(['neg', 'pos']):
            label_dir = os.path.join(base, label_name)
            if not os.path.exists(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.endswith('.txt'):
                    with open(os.path.join(label_dir, fname), 'r', encoding='utf-8',
                              errors='replace') as f:
                        samples.append((f.read(), label_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Random / synthetic datasets ───────────────────────────────────────────────

class RandomClassificationDataset(Dataset):
    """Synthetic dataset for quick testing."""

    def __init__(self, n_samples: int = 1000, n_features: int = 20, n_classes: int = 2,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        self.y = rng.integers(0, n_classes, n_samples)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


class RandomImageDataset(Dataset):
    """Synthetic image dataset for quick testing."""

    def __init__(self, n_samples: int = 100, channels: int = 3, height: int = 32,
                 width: int = 32, n_classes: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((n_samples, channels, height, width)).astype(np.float32)
        self.y = rng.integers(0, n_classes, n_samples)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


# ── Train/val split ───────────────────────────────────────────────────────────

def random_split(
    dataset: Dataset,
    lengths: Sequence[Union[int, float]],
    seed: Optional[int] = None,
) -> List[Subset]:
    """Split dataset into non-overlapping subsets of given lengths."""
    n = len(dataset)
    # Convert floats to ints
    if any(isinstance(l, float) for l in lengths):
        lengths = [int(l * n) if isinstance(l, float) else l for l in lengths]
        # Adjust last to avoid rounding errors
        diff = n - sum(lengths)
        lengths[-1] += diff

    indices = np.arange(n)
    if seed is not None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    subsets = []
    offset = 0
    for length in lengths:
        subsets.append(Subset(dataset, indices[offset:offset + length].tolist()))
        offset += length
    return subsets


__all__ = [
    'Dataset', 'TensorDataset', 'Subset', 'ConcatDataset',
    'Sampler', 'SequentialSampler', 'RandomSampler',
    'WeightedRandomSampler', 'DistributedSampler',
    'DataLoader',
    'MNIST', 'CIFAR10', 'TinyShakespeare', 'IMDb',
    'RandomClassificationDataset', 'RandomImageDataset',
    'random_split',
]
