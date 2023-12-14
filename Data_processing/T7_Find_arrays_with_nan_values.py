import numpy as np
import pandas as pd

from Utils import utils

config = utils.read_config()
system = 0 # 1 or 2
if system == 1:
    source_path = config["Source"]["paths"]["source_path_system_1"]
elif system == 2:
    source_path = config["Source"]["paths"]["source_path_system_2"]
else:
    source_path = ""
    print("Invalid system")
    
# The following part of the code it is used to find arrays that might have NaN values
df_of_collages = pd.read_excel(source_path + config["reshaped_collages_dataframe"])
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
df_with_nan_arrays.to_excel(source_path + config["collages_with_nan_arrays_daraframe"], index=False)
df_of_collages_without_nan_arrays = temp[~temp.incorrect_projection.isin(df_with_nan_arrays.incorrect_projection)]
df_of_collages_without_nan_arrays.drop(columns=['incorrect_projection'])
df_of_collages_without_nan_arrays.to_excel(source_path + config["collages_without_nan_arrays_daraframe"], index=False)
