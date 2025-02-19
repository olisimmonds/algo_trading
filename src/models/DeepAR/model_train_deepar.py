import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import torch
# from gluonts.torch.model.deepar import DeepAREstimator
# from gluonts.torch. import Trainer
# from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.torch import DeepAREstimator
# estimator = SimpleFeedForwardEstimator(prediction_length=6)

# # Open and read the JSONL file
# with open("deepar_train_data.jsonl", "r") as file:
#     training_data = [json.loads(line) for line in file]

# predictor = estimator.train(training_data[0])

# hyperparameters = {
#     # we use 5 min frequency for the time series
#     "time_freq": "5m",
#     # we predict for 30 mins
#     "prediction_length": 30 // 5,
#     # we use a context of 3 hours
#     "context_length": 180 // 5,
#     "epochs": 400,
#     "early_stopping_patience": 40,
#     "mini_batch_size": 64,
#     "learning_rate": 5E-4,
#     "num_hidden_dimensions": [10]
# }

# estimator = SimpleFeedForwardEstimator(prediction_length=6)
    # freq=hyperparameters["time_freq"],
    # num_hidden_dimensions=hyperparameters["num_hidden_dimensions"],
     #hyperparameters["prediction_length"],
    # context_length=hyperparameters["context_length"]
    # trainer=Trainer(
    #     ctx=torch.device("cpu"),
    #     epochs=hyperparameters["epochs"],
    #     learning_rate=hyperparameters["learning_rate"],
    #     batch_size=hyperparameters["mini_batch_size"],
    #     early_stopping_patience=hyperparameters["early_stopping_patience"],
    # ),
# )

# Path to file in the parent directory
# file_path = os.path.join("..", "..", "..", "deepar_train_data.jsonl")
# file_path = '../deepar_train_data.jsonl'

# # Open and read the JSONL file
# with open("deepar_train_data.jsonl", "r") as file:
#     training_data = [json.loads(line) for line in file]
# # print(training_data[0])
# predictor = estimator.train(training_data[0])
