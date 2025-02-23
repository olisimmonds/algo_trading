# To make predictions on future datasets, I will have to manually add 6 time unitis to the array.
# Then the evaluation function will cut of this last bit and make predictions on it.

# Need to install gluonts 0.16.0 

import json
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np # Requiers Version: 1.23.5
np.bool = np.bool_
import mxnet as mx

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from pathlib import Path
import heapq

def normalize_relative(series, dynamic = False):
    """
    Normalize a time series to start at 1 and represent changes as multiples of the initial value.
    
    Args:
        series: The original time series.
    
    Returns:
        Normalized series starting at 1.
    """

    series = np.array(series)
    if dynamic:
        if series[0][0] == 0:
            return [series[0] / series[0][1]]
        return [series[0] / series[0][0]]
    else:
        prod = series / series[0]
        return prod

# Load JSONL file
jsonl_file = "deepar_test_data.jsonl"

test_data = []
with open(jsonl_file, "r") as f:
    for line in f:
        test_data.append(json.loads(line))

# Convert to ListDataset format
test_dataset = ListDataset(
    [{
        "target": normalize_relative(entry["target"]), 
        "start": pd.Timestamp(entry["start"]), 
        "dynamic_feat": normalize_relative(entry["dynamic_feat"], dynamic = True),
        "item_id": entry["item_id"]
        } for entry in test_data],
    freq="5min"
)

# loads it back
predictor_deserialized = Predictor.deserialize(Path(""))

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_dataset,  # test dataset
    predictor=predictor_deserialized,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)

def plot_forecast(true_vals, predictions):
    plt.figure(figsize=(8, 5)) 
    plt.plot(true_vals.to_timestamp())
    forecast_entry = predictions
    forecast_entry.plot(show_label=True)
    plt.legend()

acc_diff_list = []
strat1 = []
strat2 = []
strat3 = []
strat4 = []
strat5 = []
strat6 = []
strat7 = []

meds_and_true_vals = []

max_i, max_i_val = -1, 0
for i in range(len(test_dataset)):
    last_val = tss[i].iloc[-7][0]
    lower_quantile_prediction = forecasts[i].quantile(0.25)[-1]
    median = forecasts[i].quantile(0.5)[-1]
    upper_quantile_prediction = forecasts[i].quantile(0.75)[-1]
    true_value = tss[i].iloc[-1][0]
    upper_dif = upper_quantile_prediction-last_val
    lower_dif = lower_quantile_prediction-last_val
    median_dif = median-last_val
    acc_diff = (true_value-last_val)/last_val

    acc_diff_list.append(acc_diff)

    meds_and_true_vals.append((median_dif, acc_diff))

    if median>max_i_val:
        max_i, max_i_val = i, median

    # strat1
    if lower_dif>=0:
        strat1.append(acc_diff)

    # strat2
    if upper_dif > -2*lower_dif or lower_dif>=0:
        strat2.append(acc_diff)
    
    # strat3
    if median_dif>0.005:
        strat3.append(acc_diff)
    
    # strat4
    if median_dif>0.0075:
        strat4.append(acc_diff)
    
    # strat5
    if median_dif>0.01:
        strat5.append(acc_diff)
    
    # strat6
    if median_dif>0.0125:
        strat6.append(acc_diff)
        
    # strat7
    if median_dif>0.015:
        strat7.append(acc_diff)

def top_x_values(meds_and_true_vals, x):
    top_x = sorted(meds_and_true_vals, key=lambda x: x[0], reverse=True)[:x]
    return [val for _, val in top_x]

strat_top_3 = top_x_values(meds_and_true_vals, 3)
strat_top_5 = top_x_values(meds_and_true_vals, 5)
strat_top_10 = top_x_values(meds_and_true_vals, 10)

print(f"Average movment of stocks {sum(acc_diff_list)/len(acc_diff_list)}")

def find_prof(strat):
    strat_name = [name for name, val in globals().items() if val is strat]
    if len(strat) > 0:
        print(f"Average profit for strat {strat_name[0] if strat_name else 'unknown'}: {sum(strat)/len(strat)}")

find_prof(strat1)
find_prof(strat2)
find_prof(strat3)
find_prof(strat4)
find_prof(strat5)
find_prof(strat6)
find_prof(strat7)
find_prof(strat_top_3)
find_prof(strat_top_5)
find_prof(strat_top_10)