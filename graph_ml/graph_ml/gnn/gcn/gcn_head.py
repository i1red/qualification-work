import tensorflow as tf
try:
    from tensorflow.keras.layers import Layer, Dense, Dropout, PReLU
    from tensorflow.keras.models import Sequential
except ImportError:
    from tensorflow.python.keras.layers import Layer, Dense, Dropout, PReLU
    from tensorflow.python.keras.models import Sequential


class GCNHead(Layer):
    def __init__(self, norm_adj_matrix: tf.Tensor, hidden_units: list[int],
                 use_activation: bool = False, dropout: float = 0.) -> None:
        super().__init__()

        self.norm_adj_matrix = norm_adj_matrix
        self.ff_layer = Sequential([Dense(units) for units in hidden_units])

        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.prelu = PReLU() if use_activation else None

    def call(self, node_features: tf.Tensor, *_) -> tf.Tensor:
        features = tf.linalg.matmul(self.norm_adj_matrix, node_features)
        features = self.ff_layer(features)

        if self.dropout is not None:
            features = self.dropout(features)
        if self.prelu is not None:
            features = self.prelu(features)

        return features
