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
CT_specifications_first_set = ["WB", "ax", "Venfas"]
CT_specifications_second_set = ["WB", "VEN", "AX"]
CT_specifications_third_set = ["Standard", "ax"]
CT_specifications_fourth_set = ["Nativ", "ax"]
CT_specifications_fifth_set = ["STANDARD", "CT", "RECON"]
CT_specifications_sixth_set = ["BONE", "AX"]
CT_specifications_seventh_set = "STD"
CT_specifications_eighth_set = ['VENFAS', 'AX']
resolutions = ["3.", "2.", "1."]

### Rules for PET
PET_specifications_first_set = ["QCFX", "M.Free"]
PET_specifications_second_set = ["VPFX", "M.Free"]
PET_specifications_third_set = "VPFX"

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
# display_full(final_df)

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
            elif all(item in final_part_of_path for item in CT_specifications_sixth_set):
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
            elif CT_specifications_seventh_set in final_part_of_path:
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
            elif all(item in final_part_of_path for item in CT_specifications_eighth_set):
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

        if final_part_of_path[1] in PET_selected_folders[final_part_of_path[0]]:
            continue
        else:
            if "QCFX" in third_part_of_path:
                if all(item in third_part_of_path for item in PET_specifications_first_set):
                    # shutil.copytree(folder_path,
                    #                 os.path.join(destination_path, second_part_of_path),
                    #                 dirs_exist_ok=True)
                    # print(folder_path)
                    selected_exams.append(folder_path)
                    PET_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                    continue
                elif "Static" not in third_part_of_path:
                    # shutil.copytree(folder_path,
                    #                 os.path.join(destination_path, second_part_of_path),
                    #                 dirs_exist_ok=True)
                    # print(folder_path)
                    selected_exams.append(folder_path)
                    PET_selected_folders[final_part_of_path[0]].append(final_part_of_path[1])
                    continue
            elif "QCFX" not in third_part_of_path:
                if all(item in third_part_of_path for item in PET_specifications_second_set):
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
                elif PET_specifications_third_set in third_part_of_path:
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
# display_full(results)
results.to_csv('non_code_related/Selected_exams_from_Raw_DCM_transf_date_20221205__06__n86_90GB.csv')
#
# print("----------------------------------------------------------------")
#
# all_exams = pd.DataFrame(final_df, columns=["directory"])
# all_exams[["source_directory", "patient_info"]] = all_exams['directory'].str.split(pat='/', n=1, expand=True)
# all_exams.loc[:, 'Date'] = all_exams['patient_info'].str.split("-").str[1]
# all_exams = all_exams.sort_values(by='Date', ascending=False)
# all_exams.reset_index(drop=True, inplace=True)
# display_full(all_exams['Date'])

# print(f"Folders selected for CT: {sorted(CT_selected_folders['CT'])}")
# print("----------------------------------------------------------------")
# print(f"Folders selected for PET: {sorted(PET_selected_folders['PT'])}")

# columns_results = set(results['Date'])
# columns_all_exams = set(all_exams['Date'])
#
# columns_not_in_both = columns_results.symmetric_difference(columns_all_exams)
# for column in columns_not_in_both:
#     if column in results['Date']:
#         print(f"Column {column} is in results but not in all_exams")
#     else:
#         print(f"Column {column} in in all_exams but not in results")
