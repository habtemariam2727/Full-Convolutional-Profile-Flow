import pandas as pd
import numpy as np

path1 = 'data/aus_data_cleaned_annual_test.csv'
path2 = 'data/aus_data_cleaned_annual_train.csv'

frame1 = pd.read_csv(path1, index_col=0)
frame2 = pd.read_csv(path2, index_col=0)

# Combine the two dataframes
frame = pd.concat([frame1, frame2], axis=0)
print(frame.head())
print(frame.shape)

# Add number of Days over the year column based on READING_DATE 0-365
def get_days_over_year(df):
    df['READING_DATE'] = pd.to_datetime(df['READING_DATE'])
    df['days_over_year'] = df['READING_DATE'].dt.dayofyear
    return df
frame = get_days_over_year(frame)

# Drop the READING_DATE column and CUSTOMER_ID column
frame = frame.drop(columns=['READING_DATE', 'CUSTOMER_ID'])
print(frame.head())
print(frame.shape)

# Save the dataframe to a csv file
frame.to_csv('data/aus_data_cleaned_annual_habt.csv', index=False)