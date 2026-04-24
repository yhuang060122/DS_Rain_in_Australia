# preprocessing.py

import pandas as pd

def feature_engineering(df):
    df = df.copy()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    df['TempRange'] = df['MaxTemp'] - df['MinTemp']
    df['HumidityDiff'] = df['Humidity3pm'] - df['Humidity9am']
    
    df['RainToday'] = df['RainToday'].map({'Yes':1, 'No':0})
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})
    
    df = df.drop(columns=['Date'])
    
    return df