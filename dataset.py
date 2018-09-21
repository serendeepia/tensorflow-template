import tensorflow as tf
import numpy as np

DATASET_GENERATOR_SIZE= 10000

def dataset_size(args):
    if args.data is None:
        # generator dataset we return a fixed number:
        return DATASET_GENERATOR_SIZE
    else:
        # file dataset, we count the number of lines without the header line
        with tf.gfile.Open(args.data, 'r') as f:
            size = len(f.readlines()) - 1
        return size


def dataset_fn(args):
    with tf.variable_scope('dataset'):
        if args.data is None:
            return generator_dataset(args)
        else:
            return file_dataset(args)


def file_dataset(args):
    dataset = tf.data.TextLineDataset([args.data])

    # skip header of the file
    dataset = dataset.skip(1)


    if args.mode == tf.estimator.ModeKeys.TRAIN:

        # repeat dataset for the number of epochs
        dataset = dataset.repeat(args.epochs)

        # adds shuffle if training
        dataset = dataset.shuffle(args.train_shuffle_size)

    # process line: map + py_func
    def _map_fn(example_serialized):
        def _parse(line):
            line_s = line.split()
            if len(line_s) != 2:
                raise Exception('The line must have only 2 numbers but it is {}'.format(line))
            x = np.float32(line_s[0])
            y = np.float32(line_s[1])
            return x, y

        inputs, outputs = tf.py_func(_parse,
                                     inp=[example_serialized],
                                     Tout=[tf.float32, tf.float32],
                                     stateful=True)
        # reshape data
        inputs = tf.reshape(inputs, [1])
        outputs = tf.reshape(outputs, [1])

        return {'inputs': inputs}, {'outputs': outputs}

    dataset = dataset.map(_map_fn)

    # batch size
    dataset = dataset.batch(args.batch_size)

    return dataset


def generator_dataset(args):
    def _generator():
        """
        Very basic generator. Two numbers are generated randomly and the output is the
        multiplication of the numbers
        """
        import random, math

        max_number = 7
        for _ in range(DATASET_GENERATOR_SIZE):
            x = np.float32(random.random() * max_number)
            y = 2 + math.sin(x)
            yield [x], [y]

    dataset = tf.data.Dataset.from_generator(_generator,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=([1], [1]))

    # repeat dataset for the number of epochs for training
    if args.mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat(args.epochs)

    # map function to set the inputs and outputs to dictionaries as in the file_dataset
    def _map_fn(inputs, outputs):
        return {'inputs': inputs}, {'outputs': outputs}

    dataset = dataset.map(_map_fn)

    # batch size
    dataset = dataset.batch(args.batch_size)

    return dataset
