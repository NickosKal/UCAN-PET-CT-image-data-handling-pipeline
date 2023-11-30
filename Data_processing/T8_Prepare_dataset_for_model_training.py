import pandas as pd

# Combine the metadata with the dataset of the collages.
df_of_collages_without_nan_arrays = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/df_of_collages_without_nan_arrays.xlsx")
metadata = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/Finalized_dataset_1805_exams_with_Age.xlsx")
metadata = metadata[["npr", "scan_date", "patient_age"]]
metadata.drop_duplicates(inplace=True)
collages_dataset_with_age = pd.merge(df_of_collages_without_nan_arrays, metadata, how="inner", left_on=["patient_ID", "scan_date"], right_on=["npr", "scan_date"], suffixes=["_l","_r"])

collages_dataset_with_age = collages_dataset_with_age.drop(columns=["npr"])
collages_dataset_with_age.to_excel("/media/andres/T7 Shield1/UCAN_project/dataset_for_model_training.xlsx", index=False)