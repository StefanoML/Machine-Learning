import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(type(housing))
print(housing.keys())
print(housing.feature_names)
print(housing.target_names)
print(housing.data.shape)