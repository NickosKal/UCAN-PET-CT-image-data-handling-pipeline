import pandas as pd
import numpy as np

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

# Combine the metadata with the dataset of the collages.
df_of_collages_without_nan_arrays = pd.read_excel(source_path + config["collages_without_nan_arrays_daraframe"])
metadata_for_regression = pd.read_excel(source_path + config["metadata"]["filenames"]["regression_dataframe"])
metadata_for_regression = metadata_for_regression[["npr", "scan_date", "patient_age"]]
metadata_for_regression.drop_duplicates(inplace=True)
collages_dataset_for_regression = pd.merge(df_of_collages_without_nan_arrays, metadata_for_regression, how="inner", left_on=["patient_ID", "scan_date"], right_on=["npr", "scan_date"], suffixes=["_l","_r"])

collages_dataset_for_regression = collages_dataset_for_regression.drop(columns=["npr"])
collages_dataset_for_regression.to_excel(source_path + config["collages_for_rergession_dataframe"], index=False)

metadata_for_classification = pd.read_excel(source_path + config["metadata"]["filenames"]["classification_dataframe"])
metadata_for_classification = metadata_for_classification[["patient_ID", "sex", "diagnosis_groups"]]
metadata_for_classification.drop_duplicates(inplace=True)
collages_dataset_for_classification = pd.merge(df_of_collages_without_nan_arrays, metadata_for_classification, how="inner", left_on=["patient_ID"], right_on=["patient_ID"], suffixes=["_l","_r"])

collages_dataset_for_classification = collages_dataset_for_classification[collages_dataset_for_classification.sex.isin(["MALE", "FEMALE"])]
collages_dataset_for_classification = collages_dataset_for_classification.replace("FEMALE", 0).replace("MALE", 1)
collages_dataset_for_classification["GT_diagnosis_label"] = np.where(collages_dataset_for_classification["diagnosis_groups"]=="C81", 0, np.where(collages_dataset_for_classification["diagnosis_groups"]=="C83", 1, 2))
collages_dataset_for_classification.to_excel(source_path + config["collages_for_classification_dataframe"], index=False)