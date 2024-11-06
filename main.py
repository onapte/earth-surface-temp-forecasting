from models import MLModel
from utils import date_to_datetime, celsius_to_fahrenheit, reduce_df

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from prettytable import PrettyTable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from statsmodels.tools.eval_measures import mse, rmse
from math import sqrt

def train_shallowML_algos(X_train, X_test, y_train, y_test):
  model_LR = MLModel(LinearRegression(), X_train, X_test, y_train, y_test)
  model_LR.fit()
  model_LR.show_test_statistics()

  model_KNN = MLModel(KNeighborsRegressor(n_neighbors=7), X_train, X_test, y_train, y_test)
  model_KNN.fit()
  model_KNN.show_test_statistics()

  model_RFR = MLModel(RandomForestRegressor(n_estimators=10, n_jobs=1), X_train, X_test, y_train, y_test)
  model_RFR.fit()
  model_RFR.show_test_statistics()

  model_SVR = MLModel(SVR(kernel='rbf'), X_train, X_test, y_train, y_test)
  model_SVR.fit()
  model_SVR.show_test_statistics()

  model_GBR = MLModel(GradientBoostingRegressor(n_estimators=10), X_train, X_test, y_train, y_test)
  model_GBR.fit()
  model_GBR.show_test_statistics()

  model_RR = MLModel(Ridge(alpha=0.5), X_train, X_test, y_train, y_test)
  model_RR.fit()
  model_RR.show_test_statistics()

  model_LAR = MLModel(Lasso(alpha=0.5), X_train, X_test, y_train, y_test)
  model_LAR.fit()
  model_LAR.show_test_statistics()

  model_ENR = MLModel(ElasticNet(alpha=0.5, l1_ratio=0.4), X_train, X_test, y_train, y_test)
  model_ENR.fit()
  model_ENR.show_test_statistics()

  table = PrettyTable()
  table.field_names = ("Model", "R2 Score", "MAE", "MSE", "RMSE")
  
  model_stats = {}
  for model in [model_LR, model_RFR, model_KNN, model_SVR, model_GBR, model_LAR, model_RR, model_ENR]:
      model_stats[model.model_name] = model.test_stats

  return table

if __name__ == '__main__':
  data_dir = '/input'

  global_temp_df = pd.read_csv(os.path.join(data_dir, 'climate-change-earth-surface-temperature-data', 'GlobalTemperatures.csv'))

  main_cols = ['LandAverageTemperature','LandMaxTemperature','LandMinTemperature','LandAndOceanAverageTemperature']
  global_temp_df[main_cols] = global_temp_df[main_cols].apply(celsius_to_fahrenheit)

  global_temp_df_new = date_to_datetime(global_temp_df)
  global_temp_df_new = global_temp_df_new.drop(['Month', 'dt'], axis=1)
  global_temp_df_new = global_temp_df_new.set_index('Year')
  global_temp_df_cleaned = global_temp_df_new[global_temp_df_new.index >=1850]
  global_temp_df_reduced = reduce_df(global_temp_df_cleaned)

  X = global_temp_df_reduced.drop('LandAndOceanAverageTemperature',axis=1)
  Y = global_temp_df_reduced['LandAndOceanAverageTemperature']

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=40)

  table = train_shallowML_algos(X_train, X_test, y_train, y_test)
  print(table)
