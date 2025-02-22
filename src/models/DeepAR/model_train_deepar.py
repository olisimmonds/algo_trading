import json
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
np.bool = np.bool_
import mxnet as mx

from gluonts.torch import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
import random

# Load JSONL file
jsonl_file = "deepar_train_data.jsonl"

data = []
with open(jsonl_file, "r") as f:
    for line in f:
        data.append(json.loads(line))

random.shuffle(data)

split_idx = int(0.8 * len(data))

# Train and test sets
train_data = data[:split_idx]
test_data = data[split_idx:]

def normalize_relative(series):
    """
    Normalize a time series to start at 1 and represent changes as multiples of the initial value.
    
    Args:
        series: The original time series.
    
    Returns:
        Normalized series starting at 1.
    """

    series = np.array(series)
    if type(series[0]) == "numpy.ndarray":
        return [series[0] / series[0][0]]
    else:
        prod = series / series[0]
        return prod

# Convert to ListDataset format
train_dataset = ListDataset(
    [{
        "target": normalize_relative(entry["target"]), 
        "start": pd.Timestamp(entry["start"]), 
        "dynamic_feat": normalize_relative(entry["dynamic_feat"])
        } for entry in train_data],
    freq="5min"
)

# Convert to ListDataset format
test_dataset = ListDataset(
    [{
        "target": normalize_relative(entry["target"]), 
        "start": pd.Timestamp(entry["start"]), 
        "dynamic_feat": normalize_relative(entry["dynamic_feat"])
        } for entry in test_data],
    freq="5min"
)

# Train the model and make predictions
model = DeepAREstimator(
    prediction_length=6, freq="5min", num_batches_per_epoch = 40, trainer_kwargs={"max_epochs": 100}
).train(train_dataset)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=train_dataset,  # test dataset
    predictor=model,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)

for i in range(10):
    plt.figure(figsize=(8, 5)) 
    plt.plot(tss[i].to_timestamp())
    forecast_entry = forecasts[i]
    forecast_entry.plot(show_label=True)
    plt.legend()
