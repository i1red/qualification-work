import tensorflow as tf

try:
    from tensorflow.keras.layers import Layer, PReLU, Activation, Dropout
    from tensorflow.keras.activations import swish, tanh
except ImportError:
    from tensorflow.python.keras.layers import Layer, PReLU, Activation, Dropout
    from tensorflow.python.keras.activations import swish, tanh

from graph_ml.gnn.gat.attention_head import AttentionHead


class MultiHeadGATLayer(Layer):
    def __init__(self, edges: list[tuple[int, int]], attention_adj_matrix_shape: tuple[int, int],
                 hidden_units: list[int], dropout: float = 0.25, add_self_features: bool = True) -> None:
        super().__init__()

        self.attention_heads = [AttentionHead(edges, attention_adj_matrix_shape,
                                              hidden_units, activation, add_self_features)
                                for activation in (PReLU(), Activation(swish), Activation(tanh))]

        self.dropout = Dropout(dropout)
        self.prelu = PReLU()

    def call(self, node_features: tf.Tensor, *_) -> tf.Tensor:
        features = tf.concat([attention_head(node_features) for attention_head in self.attention_heads], axis=1)
        features = self.dropout(features)
        return self.prelu(features)

