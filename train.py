import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.training.distribute import DistributionStrategy
from tensorflow.contrib.distribute import MirroredStrategy, TPUStrategy
from time import time

from estimator import CustomEstimator
from dataset import dataset_fn, dataset_size
from model import model_fn
from metrics import metrics_fn, log_fn
from loss import loss_fn
from optimizer import optimizer_fn
import time_utils


def get_available_gpus():
    """
    :return: The number of available GPUs in the current machine .
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_tpus():
    """
    :return: The number of available TPUs in the current machine .
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'TPU']


def get_training_strategy():
    """
    :return: A training strategy. It could be TPUStrategy if we are running the code in TPUs,
    DistributionStrategy if the code is distributed or MirroredStrategy if the code is running
    in one machine with more than one GPU. This code expects all the machines to have the same
    number of GPUs/TUPs.
    """
    is_tpu = len(get_available_tpus()) > 0
    if is_tpu:
        return TPUStrategy()

    multi_gpu = len(get_available_gpus()) > 1
    distributed = 'TF_CONFIG' in os.environ
    if multi_gpu and distributed:
        # TODO in TF 1.11 set it to tf.contrib.distribute.MultiWorkerMirroredStrategy()
        # return MultiWorkerMirroredStrategy()
        return DistributionStrategy
    if distributed:
        return DistributionStrategy()
    if multi_gpu:
        return MirroredStrategy()

    return None


def calculate_steps_per_epoch(args, config):
    size = dataset_size(args)
    count_one_tower = int(float(size) / args.batch_size + 0.5)
    gpus_per_node = len(get_available_gpus())
    if gpus_per_node > 1 and config.train_distribute.__class__.__name__ is 'DistributionStrategy':
        gpus_per_node = 1
    if gpus_per_node == 0:
        # if we don't have GPU we count 1 for the CPUs
        gpus_per_node = 1
    return count_one_tower / (gpus_per_node * config.num_worker_replicas)


def train(args):
    outputs_dir = args.outputs_dir
    if not tf.gfile.Exists(outputs_dir):
        tf.gfile.MakeDirs(outputs_dir)

    config = tf.estimator.RunConfig(
            model_dir=args.outputs_dir,
            tf_random_seed=args.random_seed,
            train_distribute=get_training_strategy(),
            log_step_count_steps=args.log_steps,
            save_summary_steps=args.log_steps,
            )

    hooks = []
    # add time hook to stop the training after some time
    if args.max_time is not None:
        hooks.append(StopAtTimeHook(args.max_time))
    ## add hook to show a log with different tensors
    hooks.append(tf.train.LoggingTensorHook(log_fn(), every_n_iter=args.log_steps))

    estimator = CustomEstimator(
            model_dir=args.outputs_dir,
            model_fn=model_fn,
            optimizer_fn=optimizer_fn,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            params=args,
            config=config,
            steps_per_epoch=calculate_steps_per_epoch(args, config)
            )
    estimator.train(input_fn=lambda: dataset_fn(args), hooks=hooks)


class StopAtTimeHook(tf.train.SessionRunHook):
    """Hook that requests stop after a specified time."""

    def __init__(self, time_running):
        """
        :param int time_running: Maximum time running
        """
        time_running_secs = time_utils.tdelta(time_running).total_seconds()
        self._end_time = time() + time_running_secs

    def after_run(self, run_context, run_values):
        if time() > self._end_time:
            run_context.request_stop()
