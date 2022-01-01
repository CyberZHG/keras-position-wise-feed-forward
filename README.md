# Keras Position-Wise Feed Forward

[![Version](https://img.shields.io/pypi/v/keras-position-wise-feed-forward.svg)](https://pypi.org/project/keras-position-wise-feed-forward/)
![License](https://img.shields.io/pypi/l/keras-position-wise-feed-forward.svg)

Implementation of position-wise feed forward layer in the paper: [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

## Install

```bash
pip install keras-position-wise-feed-forward
```

## Usage

```python
from tensorflow import keras
from keras_position_wise_feed_forward import FeedForward

input_layer = keras.layers.Input(shape=(None, 32))
feed_forward_layer = FeedForward(units=128)(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=feed_forward_layer)
model.compile(optimizer='adam', loss='mse')
model.summary()
```
