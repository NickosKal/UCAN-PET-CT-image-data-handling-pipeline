import shutil
from collections import defaultdict
import pandas as pd
import os
import re

"""
Logic programming to select specific CT and PET images.
"""
source_path = r"D:\U-CAN-Lymfom_A\Raw_DCM_transf_date_20221205__06__n86_90GB/"
# source_path = "Selected_for_Sorting_test/"
destination_path = r"D:\U-CAN-Lymfom_A\Selected_for_UCAN_project"


def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)


try:
    os.makedirs(destination_path, exist_ok=True)
except OSError as error:
    print(error)

### Rules for CT
CT_ignore_folders = ["bone", "lung"]
CT_specifications_first_set = ["wb", "ax", "venfas"]
CT_specifications_second_set = ["wb", "ven", "ax"]
CT_specifications_third_set = ["standard", "ax"]
CT_specifications_fourth_set = ["standard", "ct", "recon"]
CT_specifications_fifth_set = ["nat", "ax"]
CT_specifications_sixth_set = "std"
CT_specifications_seventh_set = ['venfas', 'ax']
resolutions = ["3.", "2.", "1."]

### Rules for PET
PET_specifications_first_set = ["qcfx", "m.free"]
PET_specifications_second_set = ["vpfx", "m.free"]
PET_specifications_third_set = "vpfx"

directory_list = list()
for root, dirs, files in os.walk(source_path, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))

df = pd.DataFrame(directory_list, columns=['directory'])
df[['source_directory', 'patient_info']] = df['directory'].str.split(pat='/', n=1, expand=True)
df[['patient_directory', 'PET-CT_info']] = df['patient_info'].str.split(pat='\\', n=1, expand=True)
df[['system', 'npr', 'scan_date']] = df['patient_directory'].str.split(pat='_|-', n=2, expand=True)
temp_df = df.groupby(['npr', 'scan_date']).apply(
    lambda x: True if x['PET-CT_info'].str.startswith('CT').any() and x['PET-CT_info'].str.startswith(
        'PT').any() else False).reset_index()
temp_df = temp_df[temp_df['0' == True]]
new_df = pd.merge(temp_df, df, how="inner", on=['npr', 'scan_date'], sort=True, suffixes=("_x", "_y"))
pre_sorted_df = new_df.dropna()
pre_sorted_df.loc[:, 'Resolutions'] = pre_sorted_df['PET-CT_info'].str.split('-').str[-1].str.extract(
    r'(\d+\.\d+)').astype(float)
pre_sorted_df["Has_QCFX"] = pre_sorted_df["PET-CT_info"].str.contains("QCFX")
pre_sorted_df["Has_Venfas"] = pre_sorted_df["PET-CT_info"].str.contains("Venfas")
pre_sorted_df["Has_VEN"] = pre_sorted_df["PET-CT_info"].str.contains("VEN")
pre_sorted_df["Has_VENFAS"] = pre_sorted_df["PET-CT_info"].str.contains("VENFAS")
pre_sorted_df["Has_Standard"] = pre_sorted_df["PET-CT_info"].str.contains("Standard")
pre_sorted_df["Has_Nativ"] = pre_sorted_df["PET-CT_info"].str.contains("Nativ")
final_df = pre_sorted_df.sort_values(by=['Has_QCFX', 'Has_Venfas', 'Has_VEN', 'Has_VENFAS',
                                         'Has_Standard', 'Has_Nativ', 'Resolutions'],
                                     ascending=[False, False, False, False, False, False, False])
final_df.reset_index(drop=True, inplace=True)
final_df = final_df.drop(columns=['Has_QCFX', 'Has_Venfas', 'Has_VEN', 'Has_VENFAS',
                                  'Has_Standard', 'Has_Nativ', 'Resolutions'])
# display_full(final_df['PET-CT_info'])

CT_selected_folders = defaultdict(list)
PET_selected_folders = defaultdict(list)
selected_exams = list()

for folder_path in final_df["directory"]:
    first_split_of_path = folder_path.split("/")
    second_part_of_path = first_split_of_path[1]
    # print(second_part_of_path)
    second_split_of_path = second_part_of_path.split("\\")
    # print(second_split_of_path[0])
    third_part_of_path = second_split_of_path[1]

    if third_part_of_path.startswith("CT-"):
        final_part_of_path = third_part_of_path.split("-")
        final_part_of_path = [string.lower() for string in final_part_of_path]
        resolution_str = final_part_of_path[-1][:7]

        try:
            exam_resolution = float(resolution_str)
        except ValueError:
            continue
        resolutions_below_one = list()
        resolutions_below_one.append(exam_resolution)

        if final_part_of_path[1] in CT_selected_folders[final_part_of_path[0]]:
            continue
        else:
            if any(ignore in final_part_of_path for ignore in CT_ignore_folders):
                continue
            else:
                if all(item in final_part_of_path for item in CT_specifications_first_set):
                    if exam_resolution == 3.000000:
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif any(resolution in resolution_str for resolution in resolutions):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                elif all(item in final_part_of_path for item in CT_specifications_second_set):
                    if exam_resolution == 3.000000:
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif any(resolution in resolution_str for resolution in resolutions):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                elif all(item in final_part_of_path for item in CT_specifications_third_set):
                    if exam_resolution == 3.00000:
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif any(resolution in resolution_str for resolution in resolutions):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                elif all(item in final_part_of_path for item in CT_specifications_fourth_set):
                    if exam_resolution == 3.000000:
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif any(resolution in resolution_str for resolution in resolutions):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                elif all(item in final_part_of_path for item in CT_specifications_fifth_set):
                    if exam_resolution == 3.000000:
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif any(resolution in resolution_str for resolution in resolutions):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                elif CT_specifications_sixth_set in final_part_of_path:
                    if exam_resolution == 3.000000:
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif any(resolution in resolution_str for resolution in resolutions):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                elif all(item in final_part_of_path for item in CT_specifications_seventh_set):
                    if exam_resolution == 3.000000:
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif any(resolution in resolution_str for resolution in resolutions):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue
                    elif min(enumerate(resolutions_below_one), key=lambda x: abs(x[1] - 1)):
                        # shutil.copytree(folder_path,
                        #                 os.path.join(destination_path, second_part_of_path),
                        #                 dirs_exist_ok=True)
                        # print(folder_path)
                        selected_exams.append(folder_path)
                        CT_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                        continue

    elif third_part_of_path.startswith("PT-"):
        final_part_of_path = third_part_of_path.split("-")
        final_part_of_path = [string.lower() for string in final_part_of_path]

        if final_part_of_path[1] in PET_selected_folders[final_part_of_path[0]]:
            continue
        else:
            if "qcfx" in final_part_of_path:
                if all(item in final_part_of_path for item in PET_specifications_first_set):
                    # shutil.copytree(folder_path,
                    #                 os.path.join(destination_path, second_part_of_path),
                    #                 dirs_exist_ok=True)
                    # print(folder_path)
                    selected_exams.append(folder_path)
                    PET_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                    continue
                elif "static" not in final_part_of_path:
                    # shutil.copytree(folder_path,
                    #                 os.path.join(destination_path, second_part_of_path),
                    #                 dirs_exist_ok=True)
                    # print(folder_path)
                    selected_exams.append(folder_path)
                    PET_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                    continue
            elif "qcfx" not in final_part_of_path:
                if all(item in final_part_of_path for item in PET_specifications_second_set):
                    # shutil.copytree(folder_path,
                    #                 os.path.join(destination_path, second_part_of_path),
                    #                 dirs_exist_ok=True)
                    # print(folder_path)
                    selected_exams.append(folder_path)
                    PET_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                    continue
                elif re.search(r'\b{}\b'.format(re.escape("AC")), third_part_of_path):
                    # shutil.copytree(folder_path,
                    #                 os.path.join(destination_path, second_part_of_path),
                    #                 dirs_exist_ok=True)
                    # print(folder_path)
                    selected_exams.append(folder_path)
                    PET_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                    continue
                elif all(item in final_part_of_path for item in PET_specifications_third_set):
                    selected_exams.append(folder_path)
                    PET_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                    continue

results = pd.DataFrame(selected_exams, columns=["directory"])
results[["source_directory", "patient_info"]] = results['directory'].str.split(pat='/', n=1, expand=True)
results[['patient_directory', 'PET-CT_info']] = results['patient_info'].str.split(pat='\\', n=1, expand=True)
results.loc[:, 'Date'] = results['patient_info'].str.split("-").str[1]
results = results.sort_values(by='Date', ascending=False)
results.reset_index(drop=True, inplace=True)
results = results.drop(columns='Date')
display_full(results['PET-CT_info'])

results.to_csv('non_code_related/Selected_exams_from_Raw_DCM_transf_date_20221205__06__n86_90GB.csv')