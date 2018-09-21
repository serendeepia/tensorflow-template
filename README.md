# TensorFlow-Template

Basic example of a TensorFlow project. The code runs over TensorFlow 1.10.

## Example

We can first we test the base model without test dataset in data.txt:

`python main.py --data data.txt test`

We train the model with the synthetic generated datset:

`python main.py train`

Finally we test our trained model with the test dataset in data.txt:

`python main.py --data data.txt test`

We can se the results in TensorBoard:

`tensorboard --logdir=./outputs`

Training can be launched with different set of parameters, here a complete example:

`python main.py --log_steps 1 --outputs_dir ./output --num_neurons 4,4,2 --epochs 10 --batch_size 2 --learning_rate 0.1 --learning_rate_decay 0.1 --train_shuffle_size 100 --max_time "2hours 30min" --random_seed 42 train`

Type `python main.py -h` for help about the parameters




