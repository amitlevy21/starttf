import sys

# Import model and loss
from starttf.examples.bosch_tlr.model import create_model
from starttf.examples.bosch_tlr.loss import create_loss

# Import utility functions for training and hyper parameter management.
from starttf.estimators.scientific_estimator import easy_train_and_evaluate
from starttf.utils.hyperparams import load_params

if __name__ == "__main__":
    # Load the hyper parameters.
    hyper_params_path = "starttf/examples/bosch_tlr/hyper_params.json"
    if len(sys.argv) > 1:
        hyper_params_path = sys.argv[1] + "/" + hyper_params_path
    hyper_params = load_params(hyper_params_path)

    # Invoke the training
    easy_train_and_evaluate(hyper_params, create_model, create_loss)
