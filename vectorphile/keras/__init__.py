"""vectorphile.keras — High-level Keras-compatible API.

Provides a TensorFlow/Keras-compatible interface for building, training,
and evaluating neural networks.

Usage::

    import vectorphile.keras as keras

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    score = model.evaluate(X_test, y_test)
"""
from vectorphile.nn.keras_model import (
    Model, Sequential, History,
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, LambdaCallback,
    CSVLogger, TensorBoard,
)
from vectorphile import nn as _nn

# ---------------------------------------------------------------------------
# layers namespace
# ---------------------------------------------------------------------------

class _LayersNamespace:
    """keras.layers namespace."""

    # Dense
    Dense = _nn.Linear
    Flatten = _nn.Flatten
    Dropout = _nn.Dropout

    # Convolutional
    Conv1D = _nn.Conv1d
    Conv2D = _nn.Conv2d
    Conv2DTranspose = _nn.ConvTranspose2d

    # Pooling
    MaxPooling2D = _nn.MaxPool2d
    AveragePooling2D = _nn.AvgPool2d
    GlobalAveragePooling2D = _nn.AdaptiveAvgPool2d
    GlobalMaxPooling2D = _nn.AdaptiveMaxPool2d

    # Normalization
    BatchNormalization = _nn.BatchNorm2d
    LayerNormalization = _nn.LayerNorm
    GroupNormalization = _nn.GroupNorm

    # Recurrent
    LSTM = _nn.LSTM
    GRU = _nn.GRU
    RNN = _nn.RNN

    # Attention / Transformer
    MultiHeadAttention = _nn.MultiheadAttention
    Embedding = _nn.Embedding

    # Activation layers
    ReLU = _nn.ReLU
    LeakyReLU = _nn.LeakyReLU
    ELU = _nn.ELU
    SELU = _nn.SELU
    Softmax = _nn.Softmax
    Sigmoid = _nn.Sigmoid
    Tanh = _nn.Tanh
    GELU = _nn.GELU

    # Regularization
    AlphaDropout = _nn.Dropout  # fallback: AlphaDropout not yet separate

    # Container
    Sequential = Sequential

    def Input(self, shape, name=None, dtype='float32'):
        """Keras Input() placeholder — returns a shape tuple for layer building."""
        return shape


layers = _LayersNamespace()

# ---------------------------------------------------------------------------
# losses namespace
# ---------------------------------------------------------------------------

class _LossesNamespace:
    MeanSquaredError = _nn.MSELoss
    MeanAbsoluteError = _nn.MAELoss
    BinaryCrossentropy = _nn.BCEWithLogitsLoss
    CategoricalCrossentropy = _nn.CrossEntropyLoss
    SparseCategoricalCrossentropy = _nn.CrossEntropyLoss
    KLDivergence = _nn.KLDivLoss
    Huber = _nn.HuberLoss

    # String aliases used by compile()
    _map = {
        'mse': 'MeanSquaredError',
        'mae': 'MeanAbsoluteError',
        'mean_squared_error': 'MeanSquaredError',
        'mean_absolute_error': 'MeanAbsoluteError',
        'binary_crossentropy': 'BinaryCrossentropy',
        'categorical_crossentropy': 'CategoricalCrossentropy',
        'sparse_categorical_crossentropy': 'SparseCategoricalCrossentropy',
    }

    def get(self, identifier):
        if isinstance(identifier, str):
            name = self._map.get(identifier, identifier)
            return getattr(self, name, _nn.MSELoss)()
        return identifier


losses = _LossesNamespace()

# ---------------------------------------------------------------------------
# optimizers namespace
# ---------------------------------------------------------------------------

class _OptimizersNamespace:
    from vectorphile import optim as _optim
    Adam = _optim.Adam
    SGD = _optim.SGD
    RMSprop = _optim.RMSprop
    Adagrad = _optim.Adagrad
    AdamW = _optim.AdamW

    _map = {
        'adam': 'Adam',
        'sgd': 'SGD',
        'rmsprop': 'RMSprop',
        'adagrad': 'Adagrad',
        'adamw': 'AdamW',
    }

    def get(self, identifier, **kwargs):
        if isinstance(identifier, str):
            from vectorphile import optim
            cls_name = self._map.get(identifier.lower(), identifier)
            cls = getattr(optim, cls_name, optim.Adam)
            return cls
        return identifier


optimizers = _OptimizersNamespace()

# ---------------------------------------------------------------------------
# metrics namespace
# ---------------------------------------------------------------------------

class _MetricsNamespace:
    from vectorphile.metrics import Accuracy, Precision, Recall, F1Score

    class Mean:
        def __init__(self):
            self._total = 0.0
            self._count = 0

        def update_state(self, values, sample_weight=None):
            vals = values.to_flat_list() if hasattr(values, 'to_flat_list') else (list(values) if hasattr(values, '__iter__') else [float(values)])
            self._total += sum(float(v) for v in vals)
            self._count += len(vals)

        def result(self):
            return self._total / max(1, self._count)

        def reset_state(self):
            self._total = 0.0
            self._count = 0

    _map = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1Score',
    }

    def get(self, identifier):
        if isinstance(identifier, str):
            name = self._map.get(identifier.lower(), identifier)
            return getattr(self, name, self.Mean)()
        return identifier


metrics = _MetricsNamespace()

# ---------------------------------------------------------------------------
# callbacks namespace
# ---------------------------------------------------------------------------

class _CallbacksNamespace:
    EarlyStopping = EarlyStopping
    ModelCheckpoint = ModelCheckpoint
    ReduceLROnPlateau = ReduceLROnPlateau
    LambdaCallback = LambdaCallback
    CSVLogger = CSVLogger
    TensorBoard = TensorBoard


callbacks = _CallbacksNamespace()

# ---------------------------------------------------------------------------
# initializers namespace
# ---------------------------------------------------------------------------

class _InitializersNamespace:
    from vectorphile.nn.init import (
        xavier_uniform_ as GlorotUniform,
        xavier_normal_ as GlorotNormal,
        kaiming_uniform_ as HeUniform,
        kaiming_normal_ as HeNormal,
        zeros_ as Zeros,
        ones_ as Ones,
    )

    def get(self, identifier):
        if isinstance(identifier, str):
            return getattr(self, identifier, self.GlorotUniform)
        return identifier


initializers = _InitializersNamespace()

# ---------------------------------------------------------------------------
# regularizers namespace
# ---------------------------------------------------------------------------

class _RegularizersNamespace:
    class L1:
        def __init__(self, l1=0.01):
            self.l1 = l1

        def __call__(self, weights):
            w = weights.to_flat_list() if hasattr(weights, 'to_flat_list') else list(weights)
            return self.l1 * sum(abs(v) for v in w)

    class L2:
        def __init__(self, l2=0.01):
            self.l2 = l2

        def __call__(self, weights):
            w = weights.to_flat_list() if hasattr(weights, 'to_flat_list') else list(weights)
            return self.l2 * sum(v*v for v in w)

    class L1L2:
        def __init__(self, l1=0.0, l2=0.01):
            self.l1 = l1
            self.l2 = l2

        def __call__(self, weights):
            w = weights.to_flat_list() if hasattr(weights, 'to_flat_list') else list(weights)
            return self.l1 * sum(abs(v) for v in w) + self.l2 * sum(v*v for v in w)

    def get(self, identifier):
        if identifier is None:
            return None
        if isinstance(identifier, str):
            return getattr(self, identifier.upper(), None)()
        return identifier


regularizers = _RegularizersNamespace()

# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------

def Input(shape, name=None, dtype='float32'):
    return shape


def load_model(path: str):
    """Load a vectorphile model from a checkpoint file."""
    from vectorphile.checkpoint import load
    state = load(path)
    return state.get('obj')


def save_model(model, path: str) -> None:
    """Save a vectorphile model to a checkpoint file."""
    from vectorphile.checkpoint import save
    save(model, path)


__all__ = [
    'Model', 'Sequential',
    'layers', 'losses', 'optimizers', 'metrics', 'callbacks',
    'initializers', 'regularizers',
    'Input', 'load_model', 'save_model',
    'History',
    'EarlyStopping', 'ReduceLROnPlateau', 'ModelCheckpoint',
    'LambdaCallback', 'CSVLogger', 'TensorBoard',
]
