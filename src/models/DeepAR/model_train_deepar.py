from __future__ import print_function

%matplotlib inline

import sys
import zipfile
from dateutil.parser import parse
import json
from random import shuffle
import random
import datetime
import os

import boto3
import s3fs
import sagemaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
# from ipywidgets import IntSlider, FloatSlider, Checkbox

# we use 5 min frequency for the time series
freq = "5m"
# we predict for 3o mins
prediction_length = 30/5
# we use a context of 3 hours
context_length = 180/5