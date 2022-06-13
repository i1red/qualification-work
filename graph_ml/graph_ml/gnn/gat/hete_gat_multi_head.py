import tensorflow as tf

from graph_ml.gnn.gat.hete_attention_head import HeteAttentionHead

try:
    from tensorflow.keras.layers import Layer, Dense, Activation, PReLU
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.activations import swish, tanh
except ImportError:
    from tensorflow.python.keras.layers import Layer, Dense, Activation, PReLU
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.activations import swish, tanh


class HeteGATMultiHead(Layer):
    def __init__(self, edges: list[tuple[int, int]],
                 attention_adj_matrix_shape: tuple[int, int], hidden_units: list[int]) -> None:
        super().__init__()

        self.attention_heads = [HeteAttentionHead(edges, attention_adj_matrix_shape, hidden_units, activation)
                                for activation in (PReLU(), Activation(swish), Activation(tanh))]

    def call(self, node_features_pair: tuple[tf.Tensor, tf.Tensor], *_) -> tf.Tensor:
        features = tf.concat([attention_head(node_features_pair) for attention_head in self.attention_heads], axis=1)
        return features
