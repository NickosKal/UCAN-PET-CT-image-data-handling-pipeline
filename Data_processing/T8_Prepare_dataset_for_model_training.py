import pandas as pd
import numpy as np

# Combine the metadata with the dataset of the collages.
df_of_collages_without_nan_arrays = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/df_of_collages_without_nan_arrays.xlsx")
metadata_for_regression = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/Finalized_dataset_1805_exams_with_Age.xlsx")
metadata_for_regression = metadata_for_regression[["npr", "scan_date", "patient_age"]]
metadata_for_regression.drop_duplicates(inplace=True)
collages_dataset_for_regression = pd.merge(df_of_collages_without_nan_arrays, metadata_for_regression, how="inner", left_on=["patient_ID", "scan_date"], right_on=["npr", "scan_date"], suffixes=["_l","_r"])

collages_dataset_for_regression = collages_dataset_for_regression.drop(columns=["npr"])
collages_dataset_for_regression.to_excel("/media/andres/T7 Shield1/UCAN_project/dataset_for_model_regression_training.xlsx", index=False)

metadata_for_classification = pd.read_excel("/media/andres/T7 Shield/dataset_for_training_366patients_baseline_scans_clinical20231129.xlsx")
metadata_for_classification = metadata_for_classification[["patient_ID", "sex", "diagnosis_groups"]]
metadata_for_classification.drop_duplicates(inplace=True)
collages_dataset_for_classification = pd.merge(df_of_collages_without_nan_arrays, metadata_for_classification, how="inner", left_on=["patient_ID"], right_on=["personReference"], suffixes=["_l","_r"])

collages_dataset_for_classification = collages_dataset_for_classification.drop(columns=["patient_ID"])
collages_dataset_for_classification = collages_dataset_for_classification[collages_dataset_for_classification.sex.isin(["MALE", "FEMALE"])]
collages_dataset_for_classification = collages_dataset_for_classification.replace("FEMALE", 0).replace("MALE", 1)
collages_dataset_for_classification["GT_diagnosis_label"] = np.where(collages_dataset_for_classification["diagnosis_group"]=="C83", 1, np.where(collages_dataset_for_classification["diagnosis_group"]=="C81", 2, 3))
collages_dataset_for_classification.to_excel("/media/andres/T7 Shield1/UCAN_project/dataset_for_model_classification_training.xlsx", index=False)