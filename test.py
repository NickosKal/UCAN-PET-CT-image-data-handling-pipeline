from collections import defaultdict
import os
import re
import sys
import time
from datetime import datetime
import psutil
import numpy as np
import pandas as pd
import concurrent.futures
import pydicom as dicom
import SimpleITK as sitk
import random

# Global path variables
# source_path = "/media/andres/T7 Shield/ucan_lymfom"
# source_path = "/media/andres/T7 Shield/U-CAN-Lymfom_A"
source_path = "D:\\ucan_lymfom"

rejected_folder_path = os.path.join(source_path, 'Rejected_exams_from_U-CAN-Lymfom.xlsx')
source_filtered_folder_path = os.path.join(source_path, 'Source_Filtered_exams_from_U-CAN-Lymfom.xlsx')
incomplete_folders_path1 = os.path.join(source_path, 'No_PTorCT_exams_from_U-CAN-Lymfom_before_selection_process.xlsx')
incomplete_folders_path2 = os.path.join(source_path, 'No_PTorCT_exams_from_U-CAN-Lymfom_after_selection_process.xlsx')
selected_folders_before_filtering = os.path.join(source_path, 'Selected_exams_before_filtering_from_U-CAN-Lymfom.xlsx')
selected_folders_after_filtering = os.path.join(source_path, 'Selected_exams_after_filtering_from_U-CAN-Lymfom.xlsx')
selected_folders_before_second_filtering = os.path.join(source_path, 'Selected_exams_before_second_filtering_from_U'
                                                                     '-CAN-Lymfom.xlsx')
selected_folders_after_second_filtering = os.path.join(source_path, 'Selected_exams_after_second_filtering_from_U-CAN'
                                                                    '-Lymfom.xlsx')
final_selected_folders = os.path.join(source_path, "FinalSelected_exams_from_U-CAN-Lymfom.xlsx")
list_of_distorted_images = os.path.join(source_path, 'Distorted_exams_from_U-CAN-Lymfom.xlsx')


def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)


# Loading the dataset
print(str(datetime.now()), ": Reading through the directory tree")
directory_list = list()
for root, dirs, files in os.walk(source_path, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))
        # print(os.path.join(root, name))

df = pd.DataFrame(directory_list, columns=['directory'])
# display_full(df.head(1))

df[['source_directory', 'patient_directory', 'PET-CT_info']] = df['directory'].str.rsplit(pat='\\', n=2, expand=True)
df[['system', 'npr', 'scan_date']] = df['patient_directory'].str.split(pat='_|-', n=2, expand=True)
temp_df = df.groupby(['npr', 'scan_date']).apply(
    lambda x: True if x['PET-CT_info'].str.startswith('CT').any() and x['PET-CT_info'].str.startswith(
        'PT').any() else False).reset_index()
# display_full(temp_df.head(2))

for folder_path in df['directory']:

    # Splitting the name of each row in order to take only the last part
    # that contains the CT and PET info we care about.
    patient = folder_path.rsplit(sep='\\', maxsplit=2)[1]
    examination_file = folder_path.rsplit(sep='\\', maxsplit=2)[-1]
    examination_file = examination_file.replace("_", "-")

    # Check if the folder has to do with CT or PET examination.
    if examination_file.startswith("CT-"):
        examination_file = examination_file.split("-")
        examination_file = [string.lower() for string in examination_file]
        # print(examination_file)
