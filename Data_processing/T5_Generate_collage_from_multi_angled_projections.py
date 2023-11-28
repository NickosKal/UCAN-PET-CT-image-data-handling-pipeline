import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import scipy.ndimage
import cv2
import os

import sys
from Utils import utils

def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)

output_path = "/media/andres/T7 Shield1/UCAN_project/collages/"
current_path = "/media/andres/T7 Shield1/UCAN_project/2D_projections/raw_projections/"
df_for_collages = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/df_of_raw_projections.xlsx")
df_for_collages.head()

df_for_collages["scan_date"] = df_for_collages["scan_date"].astype(str)
df_for_collages["unique_pat_ID_scan_date"] = df_for_collages["patient_ID"] + "_" + df_for_collages["scan_date"]
unique_patient = np.unique(df_for_collages["unique_pat_ID_scan_date"])

for scan_date in tqdm(unique_patient):
    temp = df_for_collages[df_for_collages["unique_pat_ID_scan_date"] == scan_date]
    save_path = os.path.join(output_path, str(temp["patient_ID"].iloc[0]), str(temp["scan_date"].iloc[0]))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    SUV_MIP_list = []
    SUV_bone_list = []
    SUV_lean_list = []
    SUV_adipose_list = []
    SUV_air_list = []

    CT_MIP_list = []
    CT_bone_list = []
    CT_lean_list = []
    CT_adipose_list = []
    CT_air_list = []

    for idx, row in temp.iterrows():
        SUV_MIP_list.append(np.load(row["SUV_MIP"]))
        SUV_bone_list.append(np.load(row["SUV_bone"]))
        SUV_lean_list.append(np.load(row["SUV_lean"]))
        SUV_adipose_list.append(np.load(row["SUV_adipose"]))
        SUV_air_list.append(np.load(row["SUV_air"]))
        
        CT_MIP_list.append(np.load(row["CT_MIP"]))
        CT_bone_list.append(np.load(row["CT_bone"]))
        CT_lean_list.append(np.load(row["CT_lean"]))
        CT_adipose_list.append(np.load(row["CT_adipose"]))
        CT_air_list.append(np.load(row["CT_air"]))
    
    SUV_MIP = np.concatenate((SUV_MIP_list[0], SUV_MIP_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_MIP.npy"), SUV_MIP)

    SUV_bone = np.concatenate((SUV_bone_list[0], SUV_bone_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_bone.npy"), SUV_bone)

    SUV_lean = np.concatenate((SUV_lean_list[0], SUV_lean_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_lean.npy"), SUV_lean)

    SUV_adipose = np.concatenate((SUV_adipose_list[0], SUV_adipose_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_adipose.npy"), SUV_adipose)

    SUV_air = np.concatenate((SUV_air_list[0], SUV_air_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_air.npy"), SUV_air)

    CT_MIP = np.concatenate((CT_MIP_list[0], CT_MIP_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_MIP.npy"), CT_MIP)

    CT_bone = np.concatenate((CT_bone_list[0], CT_bone_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_bone.npy"), CT_bone)

    CT_lean = np.concatenate((CT_lean_list[0], CT_lean_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_lean.npy"), CT_lean)

    CT_adipose = np.concatenate((CT_adipose_list[0], CT_adipose_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_adipose.npy"), CT_adipose)

    CT_air = np.concatenate((CT_air_list[0], CT_air_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_air.npy"), CT_air)

# Creating a dataset with the paths of collages and adding a column with the age
columns_to_replace = ['SUV_MIP', 'CT_MIP', 'SUV_bone', 'CT_bone', 'SUV_lean', 'CT_lean', 'SUV_adipose', 'CT_adipose', 'SUV_air', 'CT_air']
for column in columns_to_replace:
    df_for_collages[column] = df_for_collages[column].str.replace(current_path, output_path).str.replace("/-90.0", "").str.replace("/0.0", "").str.strip()

df_for_collages.drop("angle", axis=1, inplace=True)
df_for_collages.drop_duplicates(inplace=True)
df_for_collages.to_excel("/media/andres/T7 Shield1/UCAN_project/collages_data_paths.xlsx", index=False)

collages_data = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/collages_data_paths.xlsx")
metadata = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/Finalized_dataset_1805_exams_with_Age.xlsx")
metadata = metadata[["npr", "scan_date", "patient_age"]]
metadata.drop_duplicates(inplace=True)
collages_dataset_with_age = pd.merge(collages_data, metadata, how="inner", left_on=["patient_ID", "scan_date"], right_on=["npr", "scan_date"], suffixes=["_l","_r"])

collages_dataset_with_age = collages_dataset_with_age.drop(columns=["npr"])
collages_dataset_with_age.to_excel("/media/andres/T7 Shield1/UCAN_project/collages_dataset_with_age.xlsx", index=False)

utils.display_full(collages_dataset_with_age.head(3))

# The following part of the code it is used to find arrays that might have NaN values
images_with_nan_values = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/collages_dataset_with_age.xlsx")
images_with_nan_values['incorrect_projection'] = 'False'

# Loop through each row of the dataset and check if there are any arrays with NaN values in them
for index, row in images_with_nan_values.iterrows():
    arr = np.load(row["SUV_MIP"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_MIP"])
        continue

    arr = np.load(row["SUV_bone"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_bone"])
        continue
        
    arr = np.load(row["SUV_lean"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_lean"])
        continue
        
    arr = np.load(row["SUV_adipose"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_adipose"])
        continue
        
    arr = np.load(row["SUV_air"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["SUV_air"])
        continue
        
    arr = np.load(row["CT_MIP"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_MIP"])
        continue
        
    arr = np.load(row["CT_bone"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_bone"])
        continue
        
    arr = np.load(row["CT_lean"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_lean"])
        continue
        
    arr = np.load(row["CT_adipose"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_adipose"])
        continue
        
    arr = np.load(row["CT_air"])
    if np.isnan(arr).any():
        images_with_nan_values.at[index, 'incorrect_projection'] = 'True'
        print(row["CT_air"])
        continue

display_full(images_with_nan_values[images_with_nan_values['incorrect_projection'] == 'True'])
images_with_nan_values.to_excel("/media/andres/T7 Shield1/UCAN_project/dataset_with_arrays_with_nan_values.xlsx", index=False)

collages_dataset_with_age = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/collages_dataset_with_age.xlsx")
dataset_for_model_training = pd.concat([images_with_nan_values, collages_dataset_with_age]).drop_duplicates(keep=False)
dataset_for_model_training.drop(columns=["incorrect_projection"])
dataset_for_model_training.to_excel("/media/andres/T7 Shield1/UCAN_project/dataset_for_model_training.xlsx")

