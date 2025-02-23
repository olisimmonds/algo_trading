import json
import pickle
from datetime import datetime
import numpy as np
import random

# Load the data
with open('data_records.pkl', 'rb') as file:
    data_records = pickle.load(file)

# Define the output file
output_file_train = "deepar_train_data.jsonl"
output_file_test = "deepar_test_data.jsonl"

# List to hold DeepAR formatted records
deepar_train = []
deepar_test = []

prediction_length = 6

# Process each date and ticker
for date, tickers_data in data_records.items():
    for ticker, time_series in tickers_data.items():
        try:
            # Use varrying lenght datasets
            # The random number is chosen so that there is at least 3 hours in the training set and 30 mins in the test
            rand_num = random.randint(40, 60)

            # Construct DeepAR formatted entry
            deepar_train_entry = {
                "start": time_series["Time"][0].strftime("%Y-%m-%d %H:%M:%S"),
                "target": [entry for entry in time_series["Close"]][:rand_num],
                "dynamic_feat": [[entry for entry in time_series["Volume"]][:rand_num]],
                "item_id": ticker
            }

            # Construct DeepAR formatted entry
            deepar_test_entry = {
                "start": time_series["Time"][rand_num].strftime("%Y-%m-%d %H:%M:%S"),
                "target": [entry for entry in time_series["Close"]][:rand_num+prediction_length],
                "dynamic_feat": [[entry for entry in time_series["Volume"]][:rand_num+prediction_length]],
                "item_id": ticker
            }

            deepar_train.append(deepar_train_entry)
            deepar_test.append(deepar_test_entry)
        except Exception as e:
            _=1

# Save to JSONL
with open(output_file_train, "w") as f:
    for record in deepar_train:
        f.write(json.dumps(record) + "\n")

with open(output_file_test, "w") as f:
    for record in deepar_test:
        f.write(json.dumps(record) + "\n")

