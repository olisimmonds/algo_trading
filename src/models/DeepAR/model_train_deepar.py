import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from gluonts.mx import SimpleFeedForwardEstimator, Trainer

hyperparameters = {
    # we use 5 min frequency for the time series
    "time_freq": "5m",
    # we predict for 3o mins
    "prediction_length": 30/5,
    # we use a context of 3 hours
    "context_length": 180/5,
    "epochs": "400",
    "early_stopping_patience": "40",
    "mini_batch_size": "64",
    "learning_rate": "5E-4",
    "num_hidden_dimensions": [10]
}


estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=hyperparameters["num_hidden_dimensions"],
    prediction_length=hyperparameters["prediction_length"],
    context_length=hyperparameters["context_length"],
    trainer=Trainer(ctx="cpu", epochs=hyperparameters["epochs"], learning_rate=hyperparameters["learning_rate"], num_batches_per_epoch=hyperparameters["mini_batch_size"]),
)

# Path to file in the parent directory
file_path = '../deepar_train_data.jsonl'

# Open and read the JSONL file
with open(file_path, "r") as file:
    training_data = [json.loads(line) for line in file]

predictor = estimator.train(training_data)