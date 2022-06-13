import tensorflow as tf
try:
    from tensorflow.keras.layers import Layer
except ImportError:
    from tensorflow.python.keras.layers import Layer

from graph_ml.utility.feed_forward_layer import FeedForwardLayer


class GCNLayer(Layer):
    def __init__(self, norm_adj_matrix: tf.Tensor, hidden_units: list[int], dropout: float = 0.25) -> None:
        super().__init__()

        self.norm_adj_matrix = norm_adj_matrix
        self.ff_layer = FeedForwardLayer(hidden_units, dropout)

    def call(self, node_features: tf.Tensor, *_) -> tf.Tensor:
        features = tf.linalg.matmul(self.norm_adj_matrix, node_features)
        return self.ff_layer(features)
