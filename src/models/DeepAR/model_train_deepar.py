import json
import pandas as pd
import matplotlib.pyplot as plt

# Need to install gluonts 0.16.0 

import numpy as np # Requiers Version: 1.23.5
np.bool = np.bool_
import mxnet as mx

from gluonts.torch import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
import random

# Load JSONL file
jsonl_file = "deepar_train_data.jsonl"

train_data = []
with open(jsonl_file, "r") as f:
    for line in f:
        train_data.append(json.loads(line))

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
        "dynamic_feat": normalize_relative(entry["dynamic_feat"]),
        "item_id": entry["item_id"]
        } for entry in train_data],
    freq="5min"
)

# Train the model and make predictions
model = DeepAREstimator(
    prediction_length=6, 
    freq="5min", 
    num_batches_per_epoch = 50, 
    trainer_kwargs={"max_epochs": 500}
).train(train_dataset)

# save the trained model in tmp/
from pathlib import Path

model.serialize(Path(""))