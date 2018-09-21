import tensorflow as tf


def loss_fn(inputs_dict, model_outputs_dict, outputs_dict, params):
    """
    :param inputs_dict: inputs from the dataset
    :param model_outputs_dict: outputs of the model (predicted values)
    :param outputs_dict: outputs from the dataset (real values)
    :param params: configuration
    :return: a tuple with the total loss and a dict with all the losses
    """
    model_predictions = model_outputs_dict['prediction']
    real_values = outputs_dict['outputs']

    with tf.variable_scope('losses'):
        loss_mse = tf.reduce_mean(tf.squared_difference(real_values, model_predictions), name='mse')

    # Add summary for TensorBoard
    tf.summary.scalar('loss/mse', loss_mse)

    # in this case the total loss is the same as loss_mse but we could have more than one loss
    loss_total = loss_mse

    return loss_total, {
        'loss_mse': loss_mse,
        }
