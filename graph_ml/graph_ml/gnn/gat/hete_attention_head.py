import tensorflow as tf

try:
    from tensorflow.keras.layers import Layer, Dense
    from tensorflow.keras.models import Sequential
except ImportError:
    from tensorflow.python.keras.layers import Layer, Dense
    from tensorflow.python.keras.models import Sequential


class HeteAttentionHead(Layer):
    def __init__(self, edges: list[tuple[int, int]],
                 attention_adj_matrix_shape: tuple[int, int], hidden_units: list[int], activation: Layer) -> None:
        super().__init__()

        self.edges = edges
        self.attention_adj_matrix_shape = attention_adj_matrix_shape
        self.cur_hidden_layer = Sequential([Dense(units) for units in hidden_units])
        self.neighbor_hidden_layer = Sequential([Dense(units) for units in hidden_units])
        self.attention_layer = Sequential([Dense(1), activation])

    def calc_attention_scores(self, edge_features_pair: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        pairwise_concat_features = tf.concat(edge_features_pair, axis=1)

        attention_scores = self.attention_layer(pairwise_concat_features)
        attention_scores = attention_scores[:, 0]

        # apply softmax normalization: exp(v) / sum(exp(u) for u in N(v))
        segment_ids = self.edges[:, 0]

        exp_attention_scores = tf.exp(attention_scores)
        exp_attention_scores_sum = tf.math.segment_sum(exp_attention_scores, segment_ids)
        norm_attention_scores = exp_attention_scores / tf.gather(exp_attention_scores_sum, segment_ids)

        return norm_attention_scores

    def call(self, node_features_pair: tuple[tf.Tensor, tf.Tensor], *_) -> tf.Tensor:
        cur_node_features, neighbor_node_features = node_features_pair
        cur_node_features = self.cur_hidden_layer(cur_node_features)
        neighbor_node_features = self.neighbor_hidden_layer(neighbor_node_features)

        # calculate attention adjacency matrix
        edge_features_pair = (tf.gather(cur_node_features, self.edges[:, 0]),
                              tf.gather(neighbor_node_features, self.edges[:, 1]))
        pairwise_attention_scores = self.calc_attention_scores(edge_features_pair)
        attention_adj_matrix = tf.scatter_nd(self.edges, pairwise_attention_scores, self.attention_adj_matrix_shape)

        return tf.linalg.matmul(attention_adj_matrix, neighbor_node_features)
