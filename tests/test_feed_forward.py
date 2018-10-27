import os
import tempfile
import random
import unittest
import keras
import numpy as np
from keras_multi_head import MultiHeadAttention
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward


class TestFeedForward(unittest.TestCase):

    @staticmethod
    def _leaky_relu(x):
        return keras.activations.relu(x, alpha=0.01)

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(1, 3),
            name='Input',
        )
        feed_forward_layer = FeedForward(
            units=4,
            activation=self._leaky_relu,
            weights=[
                np.asarray([
                    [0.1, 0.2, 0.3, 0.4],
                    [-0.1, 0.2, -0.3, 0.4],
                    [0.1, -0.2, 0.3, -0.4],
                ]),
                np.asarray([
                    0.0, -0.1, 0.2, -0.3,
                ]),
                np.asarray([
                    [0.1, 0.2, 0.3],
                    [-0.1, 0.2, -0.3],
                    [0.1, -0.2, 0.3],
                    [-0.1, 0.2, 0.3],
                ]),
                np.asarray([
                    0.0, 0.1, -0.2,
                ]),
            ],
            name='FeedForward',
        )(input_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=feed_forward_layer,
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()
        inputs = np.array([[[0.2, 0.1, 0.3]]])
        predict = model.predict(inputs)
        expected = np.asarray([[[0.0364, 0.0432, -0.0926]]])
        self.assertTrue(np.allclose(expected, predict), predict)

    def test_fit(self):
        input_layer = keras.layers.Input(
            shape=(1, 3),
            name='Input',
        )
        att_layer = MultiHeadAttention(
            head_num=3,
            activation=self._leaky_relu,
            name='Multi-Head-Attention-1'
        )(input_layer)
        normal_layer = LayerNormalization(
            name='Layer-Normalization-1',
        )(att_layer)
        feed_forward_layer = FeedForward(
            units=12,
            activation=self._leaky_relu,
            name='FeedForward',
        )(normal_layer)
        normal_layer = LayerNormalization(
            name='Layer-Normalization-2',
        )(feed_forward_layer)
        output_layer = keras.layers.Add(name='Add')([input_layer, normal_layer])
        model = keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )

        def _generator(batch_size=32):
            while True:
                batch_inputs = np.random.random((batch_size, 1, 3))
                batch_outputs = batch_inputs + 0.2
                yield batch_inputs, batch_outputs

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=1000,
            epochs=10,
            validation_data=_generator(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            ],
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_feed_forward_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={
                '_leaky_relu': self._leaky_relu,
                'MultiHeadAttention': MultiHeadAttention,
                'LayerNormalization': LayerNormalization,
                'FeedForward': FeedForward,
            },
        )
        for inputs, _ in _generator(batch_size=3):
            predicts = model.predict(inputs)
            expect = inputs + 0.2
            for i in range(3):
                for j in range(3):
                    self.assertTrue(np.abs(expect[i, 0, j] - predicts[i, 0, j]) < 0.1, (expect, predicts))
            break
