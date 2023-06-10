import keras
import keras_tuner

from typing import Tuple, List


class Model:
    def __init__(self,
                 input_shape: Tuple,
                 optimizer: str = 'adam',
                 loss=None,
                 metrics=None,
                 ):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = list(metrics) if not isinstance(metrics, list) else metrics

        self.model = None

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return self.model


class CNN1D(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self,
              n_layers: int,
              n_filters: int,
              kernel_size: int,
              num_last_neurons: int = 1,
              activation: str = 'sigmoid',
              regularization: dict = {},  # NOQA
              ):
        input_layer = keras.layers.Input(self.input_shape)
        x = input_layer
        for _ in range(n_layers):
            x = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', **regularization)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        gap = keras.layers.GlobalAveragePooling1D()(x)
        output_layer = keras.layers.Dense(num_last_neurons, activation=activation)(gap)
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        return self.model

    def build_hyper(self, hp):
        n_layers = hp.Int('n_layers', 1, 5)
        n_filters = hp.Int('filters', min_value=32, max_value=256, step=32)
        kernel_size = hp.Int('kernel_size', min_value=10, max_value=100, step=10)

        self.build(
            n_layers=n_layers,
            n_filters=n_filters,
            kernel_size=kernel_size)
        self.compile()
        return self.model


class Transformer(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def transformer_encoder(inputs: keras.Input, head_size: int, num_heads: int, ff_dim: int, dropout: float = 0):
        # Attention and Normalization
        x = keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build(
            self,
            head_size: int,
            num_heads: int,
            ff_dim: int,
            num_transformer_blocks: int,
            mlp_units: List,
            dropout: float = 0,
            mlp_dropout: float = 0,
            n_classes: int = 2,
    ):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = keras.layers.Dense(dim, activation="relu")(x)
            x = keras.layers.Dropout(mlp_dropout)(x)
        outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
        self.model = keras.Model(inputs, outputs)
        return self.model


class HyperModel(keras_tuner.HyperModel):
    def __init__(self,
                 model,
                 dataset,
                 batch_size):
        self.build = model.build_hyper
        self.input_shape = model.input_shape
        self.dataset = dataset.dataset
        self.batch_size = batch_size

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            x=self.dataset['x_train'],
            y=self.dataset['y_train'],
            validation_data=(self.dataset['x_val'], self.dataset['y_val']),
            batch_size=self.batch_size,
            verbose=0,
            *args,
            **kwargs,
        )
