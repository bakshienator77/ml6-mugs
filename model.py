#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""

import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def get_training_steps():
    """Returns the number of batches that will be used to train the CNN.
    It is recommended to change this value."""
    return 500


def get_batch_size():
    """Returns the batch size that will be used by the CNN.
    It is recommended to change this value."""
    return 100


def cnn(features, labels, mode):
    """Returns an EstimatorSpec that is constructed using a CNN that you have to write below."""
    # Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 64, 64, 3]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    P = 0
    S = 1
    fh1 = 13
    fw1 = 13
    w1 = 64
    h1 = 64

    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      strides=S,
      kernel_size=[fh1, fw1],
      padding="valid",
      activation=tf.nn.relu)
    
    w2 = ((w1-fw1+2*P )/S)+1
    h2 = ((h1-fh1+2*P )/S)+1
    w2 = int(w2)
    h2 = int(h2)
    print(w2, h2)
#    w2 /=2
#    h2 /=2
    fh2 = 8
    fw2 = 8
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
#    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 32, 32]
    # Output Tensor Shape: [batch_size, 32, 32, 64]
    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[fh2, fw2],
      padding="valid",
      activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 32, 32, 64]
    # Output Tensor Shape: [batch_size, 16, 16, 64]
#    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    w3 = ((w2-fw2+2*P )/S)+1
    h3 = ((h2-fh2+2*P )/S)+1
    w3 = int(w3)
    h3 = int(h3)
    print(w3, h3)
#    w3 /= 2
#    h3 /= 2
    fh3 = 5
    fw3 = 5
    conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=32,
      kernel_size=[fh3, fw3],
      padding="valid",
      activation=tf.nn.relu)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 16, 16, 64]
    # Output Tensor Shape: [batch_size, 16 * 16 * 64]
    w4 = ((w3-fw3+2*P )/S)+1
    h4 = ((h3-fh3+2*P )/S)+1
    w4 = int(w4)
    h4 = int(h4)
    print(w4, h4)


    pool2_flat = tf.reshape(conv3, [-1, w4 * h4 * 32])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 16 * 16 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 4]
    logits = tf.layers.dense(inputs=dropout, units=4)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        # TODO: return tf.estimator.EstimatorSpec with prediction values of all classes

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        # TODO: Let the model train here
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        # The classes variable below exists of an tensor that contains all the predicted classes in a batch
        # TODO: eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)}
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
