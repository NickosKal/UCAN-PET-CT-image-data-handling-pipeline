import os
import sys
import pandas as pd
import numpy as np

parent_dir = os.path.abspath('../')
if "UCAN-PET-CT-image-data-handling-pipeline" not in parent_dir:
    parent_dir = os.path.abspath('./')

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print("parent_dir: ", parent_dir)

from Utils import utils

config = utils.read_config()
system = 1 # 1 or 2
if system == 1:
    source_path = config["Source"]["paths"]["source_path_system_1"]
elif system == 2:
    source_path = config["Source"]["paths"]["source_path_system_2"]
else:
    source_path = ""
    print("Invalid system")

# Combine the metadata with the dataset of the collages.
df_of_collages_without_nan_arrays = pd.read_excel(source_path + config["collages_without_nan_arrays_daraframe"])

#created new column since patient id and scan_date column was missing
df_of_collages_without_nan_arrays[['patient_ID', 'scan_date']] = df_of_collages_without_nan_arrays.unique_patient_ID_scan_date.str.split(pat='_', n=1, expand=True)
df_of_collages_without_nan_arrays['scan_date'] = df_of_collages_without_nan_arrays['scan_date'].astype(int)

metadata_for_regression = pd.read_excel(source_path + config["linked"]["filenames"]["regression_dataframe"])
metadata_for_regression = metadata_for_regression[["npr", "scan_date", "patient_age"]]
metadata_for_regression.drop_duplicates(inplace=True)
collages_dataset_for_regression = pd.merge(df_of_collages_without_nan_arrays, metadata_for_regression, how="inner", left_on=["patient_ID", "scan_date"], right_on=["npr", "scan_date"], suffixes=["_l","_r"])

collages_dataset_for_regression = collages_dataset_for_regression.drop(columns=["npr"])
#collages_dataset_for_regression.to_excel(source_path + config["collages_for_rergession_dataframe"], index=False)

metadata_for_classification = pd.read_excel(source_path + config["linked"]["filenames"]["classification_dataframe"])

#new diagnosis group for classification
metadata_for_classification["diagnosis_groups_new"] = metadata_for_classification["diagnosis"].apply(lambda x: x[:5])

metadata_for_classification = metadata_for_classification[["patient_ID", "sex", "diagnosis_groups", "diagnosis_groups_new"]]
metadata_for_classification.drop_duplicates(inplace=True)

#print(metadata_for_classification.diagnosis_groups_new.value_counts())

collages_dataset_for_classification = pd.merge(df_of_collages_without_nan_arrays, metadata_for_classification, how="inner", left_on=["patient_ID"], right_on=["patient_ID"], suffixes=["_l","_r"])

collages_dataset_for_classification = collages_dataset_for_classification[collages_dataset_for_classification.sex.isin(["MALE", "FEMALE"])]
collages_dataset_for_classification = collages_dataset_for_classification.replace("FEMALE", 0).replace("MALE", 1)
collages_dataset_for_classification["GT_diagnosis_label"] = np.where(collages_dataset_for_classification["diagnosis_groups"]=="C81", 0, np.where(collages_dataset_for_classification["diagnosis_groups"]=="C83", 1, 2))

#collages_dataset_for_classification.to_excel(source_path + config["collages_for_classification_dataframe"], index=False)

#copy dataframe
collages_dataset_for_classification2 = collages_dataset_for_classification.copy()

#print(collages_dataset_for_classification2.diagnosis_groups_new.value_counts())

#selected dignosis
#C83.3 - Diffust storcelligt B-cellslymfom
#C81.1 - Hodgkins lymfom (klassiskt) med nodulär skleros
#C81.9 - Hodgkins lymfom, ospecificerat

selected_diagnosis_list = ["C83.3", "C81.1", "C81.9"]

collages_dataset_for_classification2 = collages_dataset_for_classification2[collages_dataset_for_classification2.diagnosis_groups_new.isin(selected_diagnosis_list)]

collages_dataset_for_classification2["GT_diagnosis_label_new"] = np.where(collages_dataset_for_classification2["diagnosis_groups_new"]=="C83.3", 0, np.where(collages_dataset_for_classification2["diagnosis_groups_new"]=="C81.1", 1, 2))

print(collages_dataset_for_classification2.diagnosis_groups_new.value_counts())

print(collages_dataset_for_classification2.GT_diagnosis_label_new.value_counts())

collages_dataset_for_classification2.to_excel(source_path + config["collages_for_classification_dataframe_new_diagnosis"], index=False)
