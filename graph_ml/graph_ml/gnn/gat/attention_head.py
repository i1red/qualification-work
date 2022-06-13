import tensorflow as tf

try:
    from tensorflow.keras.layers import Layer, Dense
    from tensorflow.keras.models import Sequential
except ImportError:
    from tensorflow.python.keras.layers import Layer, Dense
    from tensorflow.python.keras.models import Sequential


class AttentionHead(Layer):
    def __init__(self, edges: list[tuple[int, int]], attention_adj_matrix_shape: tuple[int, int],
                 hidden_units: list[int], activation: Layer, add_self_features: bool = False) -> None:
        super().__init__()

        self.edges = edges
        self.attention_adj_matrix_shape = attention_adj_matrix_shape
        self.hidden_layer = Sequential([Dense(units) for units in hidden_units])
        self.attention_layer = Sequential([Dense(1), activation])

        self.add_self_features = add_self_features

    def calc_attention_scores(self, pairwise_features: tf.Tensor) -> tf.Tensor:
        pairwise_concat_features = tf.concat([pairwise_features[:, 0], pairwise_features[:, 1]], axis=1)

        attention_scores = self.attention_layer(pairwise_concat_features)
        attention_scores = attention_scores[:, 0]

        # apply softmax normalization: exp(v) / sum(exp(u) for u in N(v))
        segment_ids = self.edges[:, 0]

        exp_attention_scores = tf.exp(attention_scores)
        exp_attention_scores_sum = tf.math.segment_sum(exp_attention_scores, segment_ids)
        norm_attention_scores = exp_attention_scores / tf.gather(exp_attention_scores_sum, segment_ids)

        return norm_attention_scores

    def call(self, node_features: tf.Tensor, *_) -> tf.Tensor:
        features = self.hidden_layer(node_features)

        # calculate attention adjancency matrix
        pairwise_features = tf.gather(features, self.edges)
        pairwise_attention_scores = self.calc_attention_scores(pairwise_features)
        attention_adj_matrix = tf.scatter_nd(self.edges, pairwise_attention_scores, self.attention_adj_matrix_shape)

        if not self.add_self_features:
            return tf.linalg.matmul(attention_adj_matrix, features)

        return features + tf.linalg.matmul(attention_adj_matrix, features)
