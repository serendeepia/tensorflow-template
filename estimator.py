import tensorflow as tf


class CustomEstimator(tf.estimator.Estimator):
    """
    Custom estimator to use the external functions we have defined
    """

    def __init__(
            self,
            model_dir=None,
            model_fn=None,
            optimizer_fn=None,
            loss_fn=None,
            metrics_fn=None,
            config=None,
            warm_start_from=None,
            params=None,
            steps_per_epoch=None,
            ):
        if params is not None and steps_per_epoch is not None:
            params.steps_per_epoch = steps_per_epoch

        def _model_fn(features, labels, mode, params):
            """
            function to create the model, this is a wrapper of our custom model_fn and other
            functions we use to create the graph
            """

            # tensor for epoch counter
            if steps_per_epoch is not None:
                global_step_float = tf.cast(tf.train.get_or_create_global_step(), dtype=tf.float32)
                steps_per_epoch_float = tf.constant(steps_per_epoch, dtype=tf.float32)
                tf.identity(global_step_float / steps_per_epoch_float, name='epoch')

            inputs_dict = features
            outputs_dict = labels
            # create the base model
            model_outputs_dict = model_fn(inputs_dict, mode, params)

            # only for when we want to export the graph or do predictions
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions=model_outputs_dict)

            loss, loss_dict = loss_fn(inputs_dict, model_outputs_dict, outputs_dict, params)

            # additional metrics to evaluate the graph, without train operator
            if mode == tf.estimator.ModeKeys.EVAL:
                metrics_ops = metrics_fn(inputs_dict, model_outputs_dict, outputs_dict,
                                         loss, loss_dict, params)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics_ops)

            assert mode == tf.estimator.ModeKeys.TRAIN

            # train op for training
            train_op = optimizer_fn(loss, loss_dict, params, steps_per_epoch)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        super(CustomEstimator, self).__init__(model_fn=_model_fn,
                                              model_dir=model_dir,
                                              config=config,
                                              warm_start_from=warm_start_from,
                                              params=params)
