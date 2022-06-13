from itertools import zip_longest
from typing import Literal, Iterable

import tensorflow as tf

try:
    from tensorflow.keras.layers import Layer, PReLU, Activation, Dropout
    from tensorflow.keras.activations import swish, tanh
except ImportError:
    from tensorflow.python.keras.layers import Layer, PReLU, Activation, Dropout
    from tensorflow.python.keras.activations import swish, tanh

from graph_ml.gnn.gcn.gcn_head import GCNHead


class HeteGCNLayer(Layer):
    def __init__(self, gcn_heads: list[GCNHead],
                 combination_func_type: Literal['sum', 'concat'] = 'sum', dropout: float = 0.25) -> None:
        super().__init__()

        self.gcn_layers = gcn_heads

        combination_function_types = {
            'sum': lambda tensors: tf.reduce_sum(tensors, axis=0),
            'concat': lambda tensors: tf.concat(tensors, axis=1)
        }
        self.combination_function = combination_function_types[combination_func_type]
        self.dropout = Dropout(dropout)
        self.prelu = PReLU()

    def call(self, node_features_list: Iterable[tf.Tensor], *_) -> tf.Tensor:
        features_list = [gcn_layer(features) for gcn_layer, features in
                         zip_longest(self.gcn_layers, node_features_list)]

        combined_features = self.combination_function(features_list)
        combined_features = self.dropout(combined_features)
        combined_features = self.prelu(combined_features)

        return combined_features
