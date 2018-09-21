import tensorflow as tf


def model_fn(inputs_dict, mode, params):
    """
    Creates a very simple model with several dense layers
    :param inputs_dict: inputs dict for the dataset
    :param mode: either tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL or
    tf.estimator.ModeKeys.PREDICT
    :param params: configuration
    :return: a dict with the outputs of the model
    """
    with tf.variable_scope('my_model'):
        net = inputs_dict['inputs']
        for layers, neurons in enumerate(params.num_neurons.split(',')):
            net = tf.layers.dense(
                    net,
                    neurons,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.initializers.variance_scaling,
                    activity_regularizer=tf.layers.batch_normalization,
                    name='fc_layer_{}_neurons_{}'.format(layers + 1, neurons),
                    )
        # output layer (one number without activation function)
        net = tf.layers.dense(
                net,
                1,
                activation=None,
                kernel_initializer=tf.initializers.variance_scaling,
                activity_regularizer=tf.layers.batch_normalization,
                name='fc_output',
                )
    return {'prediction': net}
