import numpy as np
import pandas as pd


# The following part of the code it is used to find arrays that might have NaN values
df_of_collages = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/df_of_collages.xlsx")
temp = df_of_collages.copy()
temp['incorrect_projection'] = 'False'

for index, row in temp.iterrows():
    arr = np.load(row["SUV_MIP"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_MIP"])
        continue

    arr = np.load(row["SUV_bone"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_bone"])
        continue
        
    arr = np.load(row["SUV_lean"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_lean"])
        continue
        
    arr = np.load(row["SUV_adipose"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_adipose"])
        continue
        
    arr = np.load(row["SUV_air"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_air"])
        continue
        
    arr = np.load(row["CT_MIP"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_MIP"])
        continue
        
    arr = np.load(row["CT_bone"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_bone"])
        continue
        
    arr = np.load(row["CT_lean"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_lean"])
        continue
        
    arr = np.load(row["CT_adipose"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_adipose"])
        continue
        
    arr = np.load(row["CT_air"])
    if np.isnan(arr).any():
        temp.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_air"])
        continue

df_with_nan_arrays = temp[temp['incorrect_projection'] == 'True']
df_with_nan_arrays = df_with_nan_arrays.drop_duplicates()
df_with_nan_arrays = df_with_nan_arrays.drop(columns=["incorrect_projection"])
df_with_nan_arrays.to_excel("/media/andres/T7 Shield1/UCAN_project/df_of_arrays_with_nan_arrays.xlsx", index=False)
df_of_collages_without_nan_arrays = df_of_collages[~df_of_collages.patient_ID.isin(df_with_nan_arrays.patient_ID)]
df_of_collages_without_nan_arrays.to_excel("/media/andres/T7 Shield1/UCAN_project/df_of_collages_without_nan_arrays.xlsx", index=False)