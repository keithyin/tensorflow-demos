from __future__ import print_function
import tensorflow as tf
import tensorflow.nn as tfnn
import numpy as np


def batch_norm_layer(x, is_train, decay=0.9, name_or_scope=None):
    """
    x: [b, emb_dim]
    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="batch_norm_layer"):
        params_shape = [1, x.shape[-1]]
        beta = tf.get_variable("beta", params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable("gamma", params_shape, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        if is_train:
            mean, variance = tfnn.moments(x, axes=[0], keep_dims=True)
            moving_mean = tf.get_variable('moving_mean', shape=params_shape, dtype=tf.float32,
                                          initializer=tf.constant_initializer(
                                              0.0, tf.float32),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', shape=params_shape, dtype=tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32),
                                              trainable=False)
            tf.add_to_collection(tf.GraphKeys.TRAIN_OP,
                                 tf.assign(moving_mean, decay * moving_mean + (1 - decay) * mean))
            tf.add_to_collection(tf.GraphKeys.TRAIN_OP,
                                 tf.assign(moving_variance, decay * moving_variance + (1 - decay) * variance))
        else:
            mean = tf.get_variable('moving_mean', shape=params_shape, dtype=tf.float32,
                                   initializer=tf.constant_initializer(
                                       0.0, tf.float32),
                                   trainable=False)
            variance = tf.get_variable('moving_variance', shape=params_shape, dtype=tf.float32,
                                       initializer=tf.constant_initializer(1.0, tf.float32),
                                       trainable=False)
        x = tfnn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
    return x


def model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    x = batch_norm_layer(features["x"], mode == tf.estimator.ModeKeys.TRAIN)
    x = tf.layers.dense(x, units=128)
    x = tf.layers.dense(x, units=64)
    pred = tf.layers.dense(x, units=1, use_bias=False)
    pred = tf.reshape(pred, shape=[-1])

    predictions = {
        "price": pred
    }

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(outputs=predictions)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        print("build Predict graph")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, predictions=pred,
                                        reduction=tf.losses.Reduction.MEAN)

    # Configure the Training Op (for TRAIN mode)
    global_step = tf.train.get_or_create_global_step()

    if mode == tf.estimator.ModeKeys.TRAIN:
        print("build train graph")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "mse": tf.metrics.mean_squared_error(labels=labels, predictions=pred)
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)


def input_fn():
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.boston_housing.load_data()

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data.astype(np.float32)},
        y=train_labels.astype(np.float32),
        batch_size=64,
        num_epochs=500,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data.astype(np.float32)},
        y=eval_labels.astype(np.float32),
        num_epochs=1,
        shuffle=False)
    return train_input_fn, eval_input_fn


if __name__ == '__main__':
    model_dir = "model_dir"

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    estimator_config = tf.estimator.RunConfig(model_dir=model_dir,
                                              save_checkpoints_steps=100)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=estimator_config)

    train_input_fn, eval_input_fn = input_fn()

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None,
                                      start_delay_secs=20,
                                      throttle_secs=5)

    tf.estimator.train_and_evaluate(train_spec=train_spec, eval_spec=eval_spec, estimator=estimator)

    # for result in mnist_classifier.predict(eval_input_fn,
    #                                        checkpoint_path="model_dir/export/best_exporter/1615971471/variables/variables"):
    #     print(result)
    #     break
    #
    # mnist_classifier.evaluate()
