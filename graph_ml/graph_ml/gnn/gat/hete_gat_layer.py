from itertools import zip_longest
from typing import Literal

import tensorflow as tf

try:
    from tensorflow.keras.layers import Layer, Dense, PReLU, Dropout
    from tensorflow.keras.models import Sequential
except ImportError:
    from tensorflow.python.keras.layers import Layer, Dense, PReLU, Dropout
    from tensorflow.python.keras.models import Sequential

from graph_ml.gnn.gat.hete_gat_multi_head import HeteGATMultiHead


class HeteGATLayer(Layer):
    def __init__(self, hete_attention_heads: list[HeteGATMultiHead],
                 combination_func_type: Literal['sum', 'concat'] = 'sum', dropout: float = 0.25) -> None:
        super().__init__()

        self.hete_attention_head_list = hete_attention_heads

        combination_function_types = {
            'sum': lambda tensors: tf.reduce_sum(tensors, axis=0),
            'concat': lambda tensors: tf.concat(tensors, axis=1)
        }
        self.combination_function = combination_function_types[combination_func_type]
        self.dropout = Dropout(dropout)
        self.prelu = PReLU()

    def call(self, node_features_list: list[tuple[tf.Tensor, tf.Tensor]], *_) -> tf.Tensor:
        features_list = [hete_attention_head(features) for hete_attention_head, features in
                         zip_longest(self.hete_attention_head_list, node_features_list)]

        combined_features = self.combination_function(features_list)
        combined_features = self.dropout(combined_features)
        combined_features = self.prelu(combined_features)

        return combined_features
