# Keras Position-Wise Feed Forward

[![Travis](https://travis-ci.org/CyberZHG/keras-position-wise-feed-forward.svg)](https://travis-ci.org/CyberZHG/keras-position-wise-feed-forward)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-position-wise-feed-forward/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-position-wise-feed-forward)

Implementation of position-wise feed forward layer in the paper: [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

## Install

```bash
pip install keras-position-wise-feed-forward
```

## Usage

```python
import keras
from keras_position_wise_feed_forward import FeedForward

input_layer = keras.layers.Input()
feed_forward_layer = FeedForward()(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=feed_forward_layer)
model.compile(optimizer='adam', loss='mse', metrics={})
model.summary()
```
