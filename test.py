from estimator import CustomEstimator
from dataset import dataset_fn
from model import model_fn
from metrics import metrics_fn
from loss import loss_fn
from optimizer import optimizer_fn


def test(args):
    estimator = CustomEstimator(
            model_dir=args.outputs_dir,
            model_fn=model_fn,
            optimizer_fn=optimizer_fn,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            params=args,
            )
    estimator.evaluate(input_fn=lambda: dataset_fn(args))
