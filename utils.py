import pandas as pd
import numpy as np
import math

def date_to_datetime(df):
    df = df.copy()
    df['dt'] = pd.to_datetime(df['dt'])
    df['Month'] = df['dt'].dt.month
    df['Year'] = df['dt'].dt.year
    return df

def celsius_to_fahrenheit(temp_cel):
    temp_fhr = (temp_cel * 1.8) + 32;
    return temp_fhr

def reduce_df(df):
  df = df.copy()
  drop_cols = ['LandMaxTemperatureUncertainty','LandAndOceanAverageTemperatureUncertainty',
                'LandAverageTemperatureUncertainty','LandMinTemperatureUncertainty']
  df = df.drop(drop_cols, axis=1)
  return df
