"""Tests for v0.5.0 new features."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

class TestDiffusion:
    def test_ddpm_scheduler_add_noise(self):
        from vectorphile.diffusion import DDPMScheduler
        s = DDPMScheduler(100)
        x0 = np.random.randn(2, 3, 8, 8).astype('float32')
        noise = np.random.randn(*x0.shape).astype('float32')
        t = np.array([0, 50])
        noisy = s.add_noise(x0, noise, t)
        assert noisy.shape == x0.shape
        assert noisy.dtype == np.float32

    def test_ddpm_scheduler_step(self):
        from vectorphile.diffusion import DDPMScheduler
        s = DDPMScheduler(100)
        s.set_timesteps(10)
        x = np.random.randn(1, 3, 8, 8).astype('float32')
        eps = np.random.randn(*x.shape).astype('float32')
        result = s.step(eps, 50, x)
        assert result.shape == x.shape

    def test_ddim_scheduler(self):
        from vectorphile.diffusion import DDIMScheduler
        s = DDIMScheduler(100, eta=0.0)
        s.set_timesteps(10)
        x = np.random.randn(1, 3, 8, 8).astype('float32')
        eps = np.random.randn(*x.shape).astype('float32')
        result = s.step(eps, 50, x)
        assert result.shape == x.shape

    def test_flow_matching(self):
        from vectorphile.diffusion import FlowMatchingScheduler
        s = FlowMatchingScheduler(100)
        x0 = np.random.randn(2, 4).astype('float32')
        noise = np.random.randn(*x0.shape).astype('float32')
        t = np.array([0.5, 0.3])
        noisy = s.add_noise(x0, noise, t)
        assert noisy.shape == x0.shape


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGeneration:
    def _make_lm(self, vocab_size=50):
        """Toy language model: always returns uniform logits."""
        def lm(x):
            from vectorphile.core.tensor import Tensor
            B = np.asarray(x._data).shape[0]
            T = np.asarray(x._data).shape[1]
            logits = np.zeros((B, T, vocab_size), dtype='float32')
            return Tensor(logits)
        return lm

    def test_top_k_filtering(self):
        from vectorphile.generation import top_k_filtering
        logits = np.random.randn(2, 100).astype('float32')
        filtered = top_k_filtering(logits, top_k=10)
        # At most top_k entries should be non-masked per row
        finite_counts = np.sum(filtered > -1e8, axis=-1)
        assert (finite_counts <= 10).all()

    def test_top_p_filtering(self):
        from vectorphile.generation import top_p_filtering
        logits = np.random.randn(1, 100).astype('float32')
        filtered = top_p_filtering(logits, top_p=0.9)
        assert filtered.shape == logits.shape

    def test_greedy_decode(self):
        from vectorphile.generation import GreedySearchDecoder, GenerationConfig
        cfg = GenerationConfig(max_new_tokens=5, eos_token_id=49)
        decoder = GreedySearchDecoder(cfg)
        lm = self._make_lm(50)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        out = decoder(lm, input_ids)
        assert out.shape[0] == 1
        assert out.shape[1] >= 4  # at least original + some new

    def test_sampling_decode(self):
        from vectorphile.generation import SamplingDecoder, GenerationConfig
        cfg = GenerationConfig(max_new_tokens=5, do_sample=True, temperature=1.0)
        decoder = SamplingDecoder(cfg)
        lm = self._make_lm(50)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        out = decoder(lm, input_ids)
        assert out.shape[0] == 1

    def test_generate_api(self):
        from vectorphile.generation import generate
        lm = self._make_lm(50)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        out = generate(lm, input_ids, max_new_tokens=3, strategy='greedy')
        assert out.shape[1] == 6  # 3 original + 3 new


# ---------------------------------------------------------------------------
# Vision transforms
# ---------------------------------------------------------------------------

class TestVisionTransforms:
    def _img(self, C=3, H=32, W=32):
        return np.random.rand(C, H, W).astype('float32')

    def test_random_crop(self):
        from vectorphile.transforms.vision import RandomCrop
        img = self._img()
        out = RandomCrop(28)(img)
        assert out.shape == (3, 28, 28)

    def test_random_hflip(self):
        from vectorphile.transforms.vision import RandomHorizontalFlip
        img = self._img()
        out = RandomHorizontalFlip(p=1.0)(img)
        assert out.shape == img.shape
        assert np.allclose(out, img[:, :, ::-1])

    def test_normalize(self):
        from vectorphile.transforms.vision import Normalize
        img = self._img()
        out = Normalize([0.5]*3, [0.5]*3)(img)
        assert out.shape == img.shape
        assert abs(out.mean()) < 1.5

    def test_color_jitter(self):
        from vectorphile.transforms.vision import ColorJitter
        img = self._img()
        out = ColorJitter(brightness=0.2, contrast=0.2)(img)
        assert out.shape == img.shape
        assert 0 <= out.min() and out.max() <= 1.0

    def test_resize(self):
        from vectorphile.transforms.vision import Resize
        img = self._img()
        out = Resize((16, 16))(img)
        assert out.shape == (3, 16, 16)

    def test_random_erasing(self):
        from vectorphile.transforms.vision import RandomErasing
        img = self._img()
        out = RandomErasing(p=1.0)(img)
        assert out.shape == img.shape

    def test_compose(self):
        from vectorphile.transforms.vision import Compose, RandomCrop, Normalize
        transform = Compose([RandomCrop(28), Normalize([0.5]*3, [0.5]*3)])
        img = self._img()
        out = transform(img)
        assert out.shape == (3, 28, 28)

    def test_gaussian_blur(self):
        from vectorphile.transforms.vision import GaussianBlur
        img = self._img()
        out = GaussianBlur(5)(img)
        assert out.shape == img.shape

    def test_to_tensor(self):
        from vectorphile.transforms.vision import ToTensor
        img = (np.random.rand(32, 32, 3) * 255).astype('uint8')
        out = ToTensor()(img)
        assert out.shape == (3, 32, 32)
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

class TestTimeSeries:
    def _series(self, n=200):
        t = np.linspace(0, 4 * np.pi, n)
        return np.sin(t) + 0.1 * np.random.randn(n)

    def test_make_windows(self):
        from vectorphile.timeseries import make_windows
        s = self._series()
        X, y = make_windows(s, window=20, horizon=5)
        assert X.shape[1] == 20
        assert y.shape[1] == 5

    def test_decompose(self):
        from vectorphile.timeseries import decompose
        s = self._series()
        trend, seasonal, residual = decompose(s, period=12)
        assert trend.shape == s.shape
        assert seasonal.shape == s.shape
        # Additive: trend + seasonal + residual ≈ series
        recon = trend + seasonal + residual
        assert np.allclose(recon, s, atol=1e-10)

    def test_arima_fit_predict(self):
        from vectorphile.timeseries import ARIMA
        s = self._series()
        model = ARIMA(p=2, d=1, q=0)
        model.fit(s)
        fc = model.predict(steps=5)
        assert len(fc) == 5
        assert np.isfinite(fc).all()

    def test_holt_winters(self):
        from vectorphile.timeseries import HoltWinters
        s = self._series()
        model = HoltWinters(alpha=0.3, beta=0.1, gamma=0.1, period=12)
        model.fit(s)
        fc = model.predict(steps=10)
        assert len(fc) == 10

    def test_adf(self):
        from vectorphile.timeseries import ADF
        s = self._series()
        result = ADF(s)
        assert 'stat' in result
        assert 'p_value' in result

    def test_metrics(self):
        from vectorphile.timeseries import mae, mse, rmse, mape
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.1, 2.1, 2.9])
        assert mae(y, yhat) < 0.2
        assert mse(y, yhat) < 0.1
        assert rmse(y, yhat) < 0.2
        assert mape(y, yhat) < 10.0


# ---------------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------------

class TestNLP:
    def test_tokenizer(self):
        from vectorphile.nlp import Tokenizer
        tok = Tokenizer()
        tokens = tok.tokenize("Hello, World!")
        assert 'hello' in tokens
        assert ',' in tokens

    def test_vocabulary(self):
        from vectorphile.nlp import Vocabulary
        vocab = Vocabulary()
        corpus = [['the', 'cat', 'sat'], ['the', 'cat', 'ate']]
        vocab.build(corpus, min_freq=1)
        ids = vocab.encode(['the', 'cat'])
        assert len(ids) == 2
        decoded = vocab.decode(ids)
        assert 'the' in decoded

    def test_tfidf(self):
        from vectorphile.nlp import TFIDFVectorizer
        corpus = ["the cat sat", "the cat ate", "dogs run fast"]
        vec = TFIDFVectorizer(max_features=100)
        X = vec.fit_transform(corpus)
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_ngram_model(self):
        from vectorphile.nlp import NGramModel
        corpus = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'],
                  ['a', 'cat', 'ran'], ['the', 'cat', 'ate']]
        model = NGramModel(n=2)
        model.fit(corpus)
        p = model.probability(['the', 'cat'])
        assert 0 <= p <= 1
        ppl = model.perplexity(corpus[:2])
        assert ppl > 0

    def test_edit_distance(self):
        from vectorphile.nlp import edit_distance
        assert edit_distance("kitten", "sitting") == 3
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "abc") == 0

    def test_bleu(self):
        from vectorphile.nlp import bleu_score
        ref = [['the', 'cat', 'sat', 'on', 'the', 'mat']]
        hyp = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        score = bleu_score(ref, hyp, max_n=4)
        assert score > 0.9  # near-perfect

    def test_rouge(self):
        from vectorphile.nlp import rouge_score
        ref = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        hyp = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        result = rouge_score(ref, hyp, n=2)
        assert result['rouge-2']['f1'] > 0.9


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------

class TestInterpret:
    def _simple_model(self):
        from vectorphile.nn import Linear, Sequential, ReLU
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        return model

    def test_saliency_map(self):
        from vectorphile.interpret import SaliencyMap
        model = self._simple_model()
        sal = SaliencyMap(model)
        x = np.random.randn(4).astype('float32')
        result = sal(x, target=0)
        assert result.shape == (4,)
        assert (result >= 0).all()

    def test_permutation_importance(self):
        from vectorphile.interpret import PermutationImportance
        model = self._simple_model()
        pi = PermutationImportance(model, n_repeats=2)
        X = np.random.randn(20, 4).astype('float32')
        y = np.random.randint(0, 2, 20)
        imp = pi(X, y)
        assert imp.shape == (4,)

    def test_lime_tabular(self):
        from vectorphile.interpret import LIME
        model = self._simple_model()
        lime = LIME(model, n_samples=50)
        x = np.random.randn(4).astype('float32')
        result = lime.explain_tabular(x, class_idx=0)
        assert 'weights' in result
        assert len(result['weights']) == 4


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

class TestPruning:
    def test_unstructured_prune(self):
        from vectorphile.nn import Linear
        from vectorphile.nn.pruning import prune_unstructured, sparsity
        model = Linear(10, 10)
        import numpy as np
        # Set dummy grad to allow sparsity calc
        prune_unstructured(model, amount=0.5, name='weight')
        w = np.asarray(model.weight._data)
        zero_frac = (w == 0).sum() / w.size
        assert zero_frac >= 0.4  # allow some tolerance

    def test_global_prune(self):
        from vectorphile.nn import Sequential, Linear, ReLU
        from vectorphile.nn.pruning import global_unstructured_prune, sparsity
        model = Sequential(Linear(8, 8), ReLU(), Linear(8, 4))
        global_unstructured_prune(model, amount=0.5)
        s = sparsity(model)
        assert s >= 0.4

    def test_gradual_pruner(self):
        from vectorphile.nn import Sequential, Linear, ReLU
        from vectorphile.nn.pruning import GradualMagnitudePruner
        model = Sequential(Linear(8, 8), ReLU(), Linear(8, 4))
        pruner = GradualMagnitudePruner(0.0, 0.8, begin_step=0, end_step=10, frequency=1)
        for _ in range(10):
            s = pruner.step(model)
        assert s >= 0.0


# ---------------------------------------------------------------------------
# Session 15: Mixtral, Falcon, Whisper, CLIP
# ---------------------------------------------------------------------------

class TestSession15Architectures:
    def test_falcon_block(self):
        import vectorphile as vp
        from vectorphile.nn.architectures import FalconBlock
        blk = FalconBlock(embed_dim=32, num_heads=4, ffn_dim=64)
        x = vp.randn(1, 4, 32)
        out = blk(x)
        assert out.shape == (1, 4, 32)

    def test_falcon_instantiate(self):
        from vectorphile.nn.architectures import Falcon
        m = Falcon(vocab_size=200, embed_dim=32, num_heads=4, num_layers=2, ffn_dim=64)
        assert m is not None
        params = list(m.parameters())
        assert len(params) > 0

    def test_mixtral_instantiate(self):
        from vectorphile.nn.architectures import Mixtral
        m = Mixtral(vocab_size=200, embed_dim=32, num_heads=4, num_kv_heads=2,
                    num_layers=2, ffn_dim=64, num_experts=4, top_k=2)
        assert m is not None

    def test_whisper_tiny_instantiate(self):
        from vectorphile.nn.architectures.whisper import Whisper
        m = Whisper.tiny()
        assert m is not None

    def test_clip_text_encoder(self):
        import vectorphile as vp
        from vectorphile.nn.architectures.clip import CLIPTextEncoder
        enc = CLIPTextEncoder(vocab_size=100, context_length=8, embed_dim=32,
                               num_heads=4, num_layers=2, ffn_dim=64, proj_dim=32)
        tokens = vp.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
        out = enc(tokens)
        assert out.shape == (1, 32)

    def test_clip_instantiate(self):
        from vectorphile.nn.architectures.clip import CLIP
        m = CLIP(vocab_size=100, context_length=8, image_size=32, patch_size=8,
                 embed_dim=32, proj_dim=32, text_num_heads=4, text_num_layers=2,
                 vision_embed_dim=64, vision_num_heads=4, vision_num_layers=2)
        assert m is not None

    def test_new_distributions_chi2_studentt(self):
        from vectorphile import random as vpr
        c = vpr.chi2(df=4, size=50)
        assert c.shape == (50,)
        cf = c._data.to_flat_list()
        assert all(v >= 0 for v in cf)
        t = vpr.student_t(df=5, size=30)
        assert t.shape == (30,)

    def test_global_optimize_basinhopping(self):
        from vectorphile.optimize import basinhopping
        result = basinhopping(lambda x: (x[0]-1.5)**2 + (x[1]+0.5)**2,
                               [0.0, 0.0], n_iter=10, stepsize=0.5)
        assert result.success
        assert abs(result.x[0] - 1.5) < 1.0

    def test_global_optimize_nnls(self):
        from vectorphile.optimize import nnls
        A = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        b = [1.0, 1.0, 2.0]
        x, residual = nnls(A, b)
        assert all(v >= -1e-9 for v in x)

    def test_sarima_forecast(self):
        from vectorphile.timeseries import SARIMA
        import math
        series = [math.sin(i*0.3) for i in range(60)]
        model = SARIMA(order=(1,1,1), seasonal_order=(1,1,1,12))
        model.fit(series)
        fc = model.forecast(5)
        assert len(fc) == 5

    def test_garch_fit(self):
        from vectorphile.timeseries import GARCH
        import random
        random.seed(42)
        returns = [random.gauss(0, 1)*0.01 for _ in range(100)]
        model = GARCH(p=1, q=1)
        model.fit(returns)
        var_fc = model.forecast_variance(returns, steps=5)
        assert len(var_fc) == 5

    def test_adaptive_max_pool1d(self):
        import vectorphile as vp
        from vectorphile.nn import functional as F
        x = vp.randn(2, 8, 16)
        out = F.adaptive_max_pool1d(x, 4)
        assert out.shape == (2, 8, 4)

    def test_adaptive_max_pool2d(self):
        import vectorphile as vp
        from vectorphile.nn import functional as F
        x = vp.randn(2, 4, 8, 8)
        out = F.adaptive_max_pool2d(x, (4, 4))
        assert out.shape == (2, 4, 4, 4)

    def test_max_pool1d_functional(self):
        import vectorphile as vp
        from vectorphile.nn import functional as F
        x = vp.randn(2, 4, 16)
        out = F.max_pool1d(x, kernel_size=4, stride=2)
        assert out.shape[0] == 2
        assert out.shape[1] == 4

    def test_min_p_filtering(self):
        import numpy as np
        from vectorphile.generation import min_p_filtering
        import vectorphile as vp
        logits = vp.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        out = min_p_filtering(logits, min_p=0.1)
        # higher logit tokens should survive, low probability tokens filtered
        assert out.shape == (1, 5)

    def test_typical_filtering(self):
        import vectorphile as vp
        from vectorphile.generation import typical_filtering
        logits = vp.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        out = typical_filtering(logits, mass=0.9)
        assert out.shape == (1, 5)

    def test_convolve_signal(self):
        import vectorphile as vp
        from vectorphile.signal import convolve
        x = vp.tensor([1.0, 2.0, 3.0, 4.0])
        h = vp.tensor([1.0, 1.0])
        out = convolve(x, h, mode='full')
        flat = out._data.to_flat_list() if hasattr(out._data, 'to_flat_list') else list(out._data)
        assert len(flat) == 5  # nx + nh - 1

    def test_linalg_svd(self):
        import vectorphile as vp
        x = vp.randn(4, 3)
        U, S, Vh = vp.linalg.svd(x, full_matrices=False)
        assert U.shape[0] == 4
        assert len(S._data.to_flat_list() if hasattr(S._data, 'to_flat_list') else list(S._data)) == 3

    def test_special_erf_erfinv(self):
        import math
        from vectorphile.special import erf, erfinv
        for v in [0.0, 0.5, -0.5]:
            assert abs(erfinv(erf(v)) - v) < 1e-5

    def test_causal_inference(self):
        from vectorphile.causal import IPWEstimator
        X = [[1.0, 0.5], [0.0, 1.0], [1.0, 0.3], [0.0, 0.8]]
        treatment = [1, 0, 1, 0]
        outcome = [2.0, 1.0, 2.5, 0.8]
        ipw = IPWEstimator()
        ipw.fit(X, treatment, outcome)
        ate = ipw.ate()
        assert isinstance(ate, float)
