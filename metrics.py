import tensorflow as tf


def metrics_fn(inputs_dict, model_outputs_dict, outputs_dict, loss, loss_dict, params):
    """
    This function is used to parse the metrics to the stimators during the evaluation

    :param inputs_dict: inputs from the dataset
    :param model_outputs_dict: outputs of the model (predicted values)
    :param outputs_dict: outputs from the dataset (real values)
    :param loss: total loss
    :param loss_dict: a dict with all the losses
    :param params: configuration
    :return: the metrics used for the evaluation
    """

    mean_squared_error, update_op = tf.metrics.mean_squared_error(
            labels=outputs_dict['outputs'],
            predictions=model_outputs_dict['prediction'],
            name='mean_square_error'
            )

    return {
        'loss/mse': (mean_squared_error, update_op)
        }


def log_fn():
    """
    This function is called in the LoggingTensorHook and it returns a dict with the tensors that
    are going to be logged.
    """
    return {
        'loss mse'     : 'losses/mse',
        'epoch'        : 'epoch',
        'learning rate': 'optimizer/learning_rate',
        }
