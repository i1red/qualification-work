import tensorflow as tf

try:
    from tensorflow.keras.layers import Layer, Dense, PReLU, Dropout
    from tensorflow.keras.models import Sequential
except ImportError:
    from tensorflow.python.keras.layers import Layer, Dense, PReLU, Dropout
    from tensorflow.python.keras.models import Sequential


class FeedForwardLayer(Layer):
    def __init__(self, hidden_units: list[int], dropout: float = 0.2) -> None:
        super().__init__()

        layers = []
        for units in hidden_units:
            layers.append(Dense(units))
            layers.append(PReLU())
        layers.append(Dropout(dropout))

        self.ff_sequential = Sequential(layers)

    def call(self, features: tf.Tensor, *_) -> tf.Tensor:
        return self.ff_sequential(features)
