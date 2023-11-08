# Logic script in order to create a dataframe that contains the exams that fit the selection rules.
# Necessary imports and source path.
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

dicom.config.convert_wrong_length_to_UN = True

# Global path variables
source_path_for_non_GPU_system = "/media/andres/T7 Shield/ucan_lymfom/Excel_files/06_11_2023/"
# source_path = "/media/andres/T7 Shield/U-CAN-Lymfom_A"
# source_path = "F:/ucan_lymfom"
#source_path = "D:\\ucan_lymfom"

# rejected_folder_path = os.path.join(source_path_for_non_GPU_system, 'Rejected_exams_from_U-CAN-Lymfom.xlsx')
# source_filtered_folder_path = os.path.join(source_path_for_non_GPU_system, 'Source_Filtered_exams_from_U-CAN-Lymfom.xlsx')
incomplete_folders_before_filtering = os.path.join(source_path_for_non_GPU_system, 'No_PT_or_CT_exams_from_U-CAN-Lymfom_before_selection_process.xlsx')
incomplete_folders_after_filtering = os.path.join(source_path_for_non_GPU_system, 'No_PT_or_CT_exams_from_U-CAN-Lymfom_after_selection_process.xlsx')
sorted_folders_before_filtering = os.path.join(source_path_for_non_GPU_system, 'Selected_exams_before_filtering_from_U-CAN-Lymfom.xlsx')
selected_folders_after_filtering = os.path.join(source_path_for_non_GPU_system, 'Selected_exams_after_filtering_from_U-CAN-Lymfom.xlsx')
final_selected_folders = os.path.join(source_path_for_non_GPU_system, "Final_Selected_exams_from_U-CAN-Lymfom.xlsx")

# Function responsible for displaying the full information of the dataframe
def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)

def data_filtering(dataframe_column):
    # Create three lists one for CT and PET folders and one for both.
    CT_selected_folders = defaultdict(list)
    PET_selected_folders = defaultdict(list)
    selected_exams = list()

    # Looping through the dataset
    for folder_path in dataframe_column:

        # Splitting the name of each row in order to take only the last part 
        # that contains the CT and PET info we care about.
        patient = folder_path.rsplit(sep='/', maxsplit=2)[1]
        examination_file = folder_path.rsplit(sep='/', maxsplit=2)[-1]
        examination_file = examination_file.replace("_", "-")

        # Check if the folder has to do with CT or PET examination.
        if examination_file.startswith("CT-"):
            examination_file = examination_file.split("-")
            examination_file = [string.lower() for string in examination_file]
            resolution_str = examination_file[-1][:-2]

            # Save the resolution to a variable for later usage.
            try:
                exam_resolution = float(resolution_str)
            except ValueError:
                continue
            resolutions_below_one = list()
            resolutions_below_one.append(exam_resolution)

            # Check if a CT examination has been already selected for the current patient.
            # If yes proceed with the next folder otherwise continue with the current one.
            if patient in CT_selected_folders[examination_file[0]]:
                continue
            else:
                # Check if the current folder contains information that we do not find useful.
                if any(ignore in examination_file for ignore in CT_ignore_folders):
                    continue
                else:
                    # The following if statements go through the set of rules that have been established in
                    # hierarchically order. Meaning that it checks if any of the folders contain information that are
                    # more important than others. The next step is to examine the resolution of the examination. Our
                    # priority is the following 3.0mm -> 2.0mm -> 1.0mm and if none of those is fulfilled then we
                    # choose the one that is closer to one by saving the path of the file to the list of the selected
                    # ones. Also, we save the CT selected list in order to be able to know that we have already
                    # selected a CT examination for the current patient.
                    if all(item in examination_file for item in CT_specifications_first_set):
                        if exam_resolution == 3.000000:
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif any(resolution in resolution_str for resolution in resolutions):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                    elif all(item in examination_file for item in CT_specifications_second_set):
                        if exam_resolution == 3.000000:
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif any(resolution in resolution_str for resolution in resolutions):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                    elif all(item in examination_file for item in CT_specifications_third_set):
                        if exam_resolution == 3.00000:
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif any(resolution in resolution_str for resolution in resolutions):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                    elif all(item in examination_file for item in CT_specifications_fourth_set):
                        if exam_resolution == 3.000000:
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif any(resolution in resolution_str for resolution in resolutions):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                    elif all(item in examination_file for item in CT_specifications_fifth_set):
                        if exam_resolution == 3.000000:
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif any(resolution in resolution_str for resolution in resolutions):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                    elif CT_specifications_sixth_set in examination_file:
                        if exam_resolution == 3.000000:
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif any(resolution in resolution_str for resolution in resolutions):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                    elif all(item in examination_file for item in CT_specifications_seventh_set):
                        if exam_resolution == 3.000000:
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif any(resolution in resolution_str for resolution in resolutions):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue
                        elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                            selected_exams.append(folder_path)
                            CT_selected_folders[examination_file[0]].append(patient)
                            continue

        # The same procedure and logic is applied for the PET folders.
        elif examination_file.startswith("PT-"):
            examination_folder = examination_file.split("-")
            examination_folder = [string.lower() for string in examination_folder]

            if patient in PET_selected_folders[examination_folder[0]]:
                continue
            else:
                if "qcfx" in examination_folder:
                    if all(item in examination_folder for item in PET_specifications_first_set):
                        selected_exams.append(folder_path)
                        PET_selected_folders[examination_folder[0]].append(patient)
                        continue
                    elif "static" not in examination_folder:
                        selected_exams.append(folder_path)
                        PET_selected_folders[examination_folder[0]].append(patient)
                        continue
                elif "qcfx" not in examination_folder:
                    if all(item in examination_folder for item in PET_specifications_second_set):
                        selected_exams.append(folder_path)
                        PET_selected_folders[examination_folder[0]].append(patient)
                        continue
                    elif re.search(r'\b{}\b'.format(re.escape("AC")), examination_file):
                        selected_exams.append(folder_path)
                        PET_selected_folders[examination_folder[0]].append(patient)
                        continue
                    elif all(item in examination_folder for item in PET_specifications_third_set):
                        selected_exams.append(folder_path)
                        PET_selected_folders[examination_folder[0]].append(patient)
                        continue
    return selected_exams

# Choose final PET and CT images if there are multiple selected images
def finalPETCT(img_lst):
    print("img_lst: ", img_lst, "\n")
    CT_lst = [img for img in img_lst if img.startswith('CT-')]

    PT_lst = [img for img in img_lst if img.startswith('PT-')]

    idx_i, idx_j = 0, 0

    if len(CT_lst) > 2:
        max = 99.0
        for i, img in enumerate(CT_lst):
            diff = abs(3.0 - float(img.split('-')[-1].replace("mm", "")))

            if diff <= max:
                idx_i = i
                max = diff

    if len(PT_lst) > 2:
        max = 99
        for j, img in enumerate(PT_lst):
            diff = abs(3.0 - float(img.split('-')[-1].replace("mm", "")))

            if diff <= max:
                idx_j = j
                max = diff
    print("idx_i: ", idx_i, "  CT_lst: ", CT_lst[idx_i])
    print("\nidx_j: ", idx_j, "  PT_lst: ", PT_lst[idx_j])

    return [CT_lst[idx_i], PT_lst[idx_j]]

if __name__ == '__main__':
    start = time.time()
    # Set of rules that affect our exam selection
    # Rules for CT
    CT_ignore_folders = ["bone", "lung", "lunga", "leg"]
    CT_specifications_first_set = ["wb", "ax", "venfas"]
    CT_specifications_second_set = ["wb", "ven", "ax"]
    CT_specifications_third_set = ["standard", "ax"]
    CT_specifications_fourth_set = ["standard", "ct", "recon"]
    CT_specifications_fifth_set = ["nat", "ax"]
    CT_specifications_sixth_set = "std"
    CT_specifications_seventh_set = ['venfas', 'ax']
    resolutions = ["3.", "2.", "1."]

    # Rules for PET
    PET_specifications_first_set = ["qcfx", "m.free"]
    PET_specifications_second_set = ["vpfx", "m.free"]
    PET_specifications_third_set = "vpfx"

    dataset = pd.read_excel(os.path.join(source_path_for_non_GPU_system, "data_ready_for_filtering.xlsx"))

    # Change the if statement in case of using a different disk
    '''
    remove_list = ['PR----BONE-PULM-mm',
                   'PR----Lunga-0.6-ax-mm',
                   'PR----WB-Venfas-0.6-ax-mm',
                   'PR----LUNG-1.25-AX-mm',
                   'PR----WB-Ben-lunga-0.6-ax-mm',
                   'PR----WB-Venfas-3-ax-mm',
                   'PR----LUNG-1.25-AX-mm',
                   'PR----BONE-1.25-AX-mm',
                   'PR----LUNG-1.25-AX-mm',
                   'PR----Lunga-0.6-ax-mm',
                   'PR----SAVED-IMAGES-PR-mm',
                   'PR----e1-QCFX-S-400-Static-mm',
                   'PR----WB-Venfas-0.6-ax-mm',
                   'PR----WB-VEN-AX-mm',
                   'PR----WB-Ben-lunga-0.6-ax-mm',
                   'PR----LUNG-1.25-AX-mm',
                   'PR----THORAX-AX-mm',
                   'PR----LUNG-1.25-AX-mm',
                   'PR----THORAX-INANDAD-mm',
                   'PR----KEY_IMAGES-PR-mm',
                   'PR----SAVED-PR-mm',
                   'Examinations that miss either CT or PET or both',
                   'MR-',
                   'sag',
                   'cor',
                   'ot-'
                   ]
                 
    keep_list = ["CT-", "PT-"]

    find_dir_lst = []
    rejection_lst = []

    for dir in directory_list:
        dir = dir.replace('\\', '/')
        if any(item.lower() in dir.lower() for item in keep_list) and all(
                item.lower() not in dir.lower() for item in remove_list):
            find_dir_lst.append(dir)
        else:
            rejection_lst.append(dir)

    print(str(datetime.now()), ': Writing rejected image folders to excel file')
    rejected_df = pd.DataFrame(rejection_lst, columns=['directory'])
    rejected_df.to_excel(rejected_folder_path)

    Creating a dataframe out of the dataset with the required information that are need to proceed with the filtering.
    print(str(datetime.now()), ": Loading the directory into Dataframe")
    df = pd.DataFrame(find_dir_lst, columns=['directory'])

    print(str(datetime.now()), ': Writing source filtered image folders to excel file')
    df.to_excel(source_filtered_folder_path)
    '''

    
    exams_with_distorted_files = pd.read_excel("/media/andres/T7 Shield/ucan_lymfom/Excel_files/06_11_2023/exams_with_distorted_images_file.xlsx")

    temporary_df = dataset[~dataset.directory.isin(exams_with_distorted_files.directory)]
    temporary_df[['source_directory', 'patient_directory', 'PET-CT_info']] = temporary_df['directory'].str.rsplit(pat='/', n=2, expand=True)

    # Uncomment to run on Windows
    # df[['source_directory', 'patient_info']] = df['directory'].str.split(pat='/', n=1, expand=True)
    # df[['patient_directory', 'PET-CT_info']] = df['patient_info'].str.split(pat='\\', n=1, expand=True)

    temporary_df[['system', 'npr', 'scan_date']] = temporary_df['patient_directory'].str.split(pat='_|-', n=2, expand=True)
    temp_df = temporary_df.groupby(['npr', 'scan_date']).apply(
        lambda x: True if x['PET-CT_info'].str.startswith('CT').any() and x['PET-CT_info'].str.startswith(
            'PT').any() else False).reset_index()

    # incomplete folders
    print(str(datetime.now()), ': Writing incomplete folders dataframe to excel')
    temp_df1 = temp_df[temp_df[0] == False].copy()
    print(str(datetime.now()), ': incomplete df shape: ', temp_df1.shape)
    incomplete_df = pd.merge(temp_df1, temporary_df, how="inner", on=['npr', 'scan_date'], sort=True, suffixes=("_x", "_y"))
    incomplete_df.to_excel(incomplete_folders_before_filtering)

    # complete folders
    print(str(datetime.now()), ': Filtering complete folders dataframe to continue execution')
    temp_df2 = temp_df[temp_df[0] == True].copy()
    print(str(datetime.now()), ': complete df shape: ', temp_df2.shape)
    new_df = pd.merge(temp_df2, temporary_df, how="inner", on=['npr', 'scan_date'], sort=True, suffixes=("_x", "_y"))
    print(str(datetime.now()), ': Shape before dropping na value: ', new_df.shape)
    pre_sorted_df = new_df.dropna()
    print(str(datetime.now()), ': Shape after dropping na value: ', pre_sorted_df.shape)
    display_full(pre_sorted_df.head(5))

    # Sort the dataset according to the rules we have in order for the most desired exams to be at the top.
    pre_sorted_df.loc[:, 'Resolutions'] = pre_sorted_df.loc[:, 'PET-CT_info'].str.split('-').str[-1].str.extract(
        r'(\d+\.\d+)').astype(float)
    pre_sorted_df.loc[:, "Has_QCFX"] = pre_sorted_df.loc[:, "PET-CT_info"].str.contains("QCFX")
    pre_sorted_df.loc[:, "Has_Venfas"] = pre_sorted_df.loc[:, "PET-CT_info"].str.contains("Venfas")
    pre_sorted_df.loc[:, "Has_VEN"] = pre_sorted_df.loc[:, "PET-CT_info"].str.contains("VEN")
    pre_sorted_df.loc[:, "Has_VENFAS"] = pre_sorted_df.loc[:, "PET-CT_info"].str.contains("VENFAS")
    pre_sorted_df.loc[:, "Has_Standard"] = pre_sorted_df.loc[:, "PET-CT_info"].str.contains("Standard")
    pre_sorted_df.loc[:, "Has_Nativ"] = pre_sorted_df.loc[:, "PET-CT_info"].str.contains("Nativ")
    final_df = pre_sorted_df.sort_values(by=['Has_QCFX', 'Has_Venfas', 'Has_VEN', 'Has_VENFAS',
                                             'Has_Standard', 'Has_Nativ', 'Resolutions'],
                                         ascending=[False, False, False, False, False, False, False])
    final_df.reset_index(drop=True, inplace=True)
    final_df = final_df.drop(columns=['Has_QCFX', 'Has_Venfas', 'Has_VEN', 'Has_VENFAS',
                                      'Has_Standard', 'Has_Nativ', 'Resolutions'])

    # # Writing the dataframe before running data filtering for selection of CT/PET images
    print(str(datetime.now()), ": Writing the dataframe before running data filtering for selection of CT/PET images")
    final_df.to_excel(sorted_folders_before_filtering)

    # Filtering the dataframe and selecting the desired exams for each patient.
    print(str(datetime.now()), ": Running the data filtering - initial run")
    selected_exams = data_filtering(final_df["directory"])
    selected_exams = pd.DataFrame(selected_exams, columns=["directory"])

    # Writing the dataframe after running first data filtering for selection of CT/PET images
    print(str(datetime.now()), ": Writing the dataframe after running data filtering for selection of CT/PET images")
    selected_exams.to_excel(selected_folders_after_filtering)

    print(str(datetime.now()), ": Number of images: ", selected_exams.shape[0])

    # Generate a dataframe with the selected examinations and saving it in the form of an Excel file.
    # Sort the dataframe by starting from the newest examination and going to the oldest.
    print(str(datetime.now()), ": Creating final dataframe having selected exams")
    selected_exams = pd.DataFrame(selected_exams, columns=["directory"])
    selected_exams[['source_directory', 'patient_directory', 'PET-CT_info']] = selected_exams['directory'].str.rsplit(
        pat='/', n=2, expand=True)
    selected_exams[['system', 'npr', 'scan_date']] = selected_exams['patient_directory'].str.split(pat='_|-', n=2,
                                                                                                   expand=True)
    selected_exams.loc[:, 'Date'] = selected_exams['patient_directory'].str.split("-").str[1]
    selected_exams = selected_exams.sort_values(by='patient_directory', ascending=False)
    selected_exams.reset_index(drop=True, inplace=True)
    selected_exams = selected_exams.drop(columns='Date')

    # Filter patient without either PET or CT
    No_PET_or_CT_agg = selected_exams.groupby(['patient_directory']).apply(
        lambda x: True if x['PET-CT_info'].str.startswith("CT-").any() and x['PET-CT_info'].str.startswith(
            "PT-").any() else False).reset_index()
    No_PET_or_CT_patient_list = No_PET_or_CT_agg[No_PET_or_CT_agg[0] == False]['patient_directory'].to_list()
    No_PET_or_CT_df = selected_exams[selected_exams["patient_directory"].isin(No_PET_or_CT_patient_list)]
    No_PET_or_CT_df.to_excel(incomplete_folders_after_filtering)

    final_results = selected_exams[~selected_exams["patient_directory"].isin(No_PET_or_CT_patient_list)]
    print("---Final Dataframe---")
    print(final_results.shape)
    display_full(final_results.head(21))
    final_temp_df = final_results.groupby(['patient_directory']).apply(
        lambda x: finalPETCT(x['PET-CT_info'].to_list())).reset_index()

    final_results = final_results[final_results["PET-CT_info"].isin(list(np.ravel(final_temp_df[0].to_list())))]
    final_results.to_excel(final_selected_folders)
    finish = time.time()
    print(f"Total time of running: {round(finish - start, 2) / 60}")
