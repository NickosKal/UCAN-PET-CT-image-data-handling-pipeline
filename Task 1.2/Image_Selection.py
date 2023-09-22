import pandas as pd
import os
import re

"""
Logic programming to select specific CT and PET images.
"""
path = "Selected_for_Sorting_test/"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 12000)

### Rules for CT
CT_specifications = ["WB", "ax", "Venfas"]
resolutions = ["1.", "2.", "3."]

### Rules for PET
PET_specifications = ["QCFX", "M.Free"]

directory_list = list()
for root, dirs, files in os.walk(path, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))

df = pd.DataFrame(directory_list, columns=['directory'])

df[['source_directory', 'patient_info']] = df['directory'].str.split(pat='/', n=1, expand=True)

df[['patient_directory', 'PET-CT_info']] = df['patient_info'].str.split(pat='\\', n=1, expand=True)

df[['system', 'npr', 'scan_date']] = df['patient_directory'].str.split(pat='_|-', n=2, expand=True)

temp_df = df.groupby(['npr', 'scan_date']).apply(
    lambda x: True if x['PET-CT_info'].str.startswith('CT').any() and x['PET-CT_info'].str.startswith(
        'PT').any() else False).reset_index()
# print(temp_df)
temp_df = temp_df[temp_df['0' == True]]

new_df = pd.merge(temp_df, df, how="inner", on=['npr', 'scan_date'], sort=True, suffixes=("_x", "_y"))

final_df = new_df.dropna()

for folder_path in final_df["directory"]:
    first_split_of_path = folder_path.split("/")
    second_part_of_path = first_split_of_path[1]
    # print(second_part_of_path)
    second_split_of_path = second_part_of_path.split("\\")
    # print(second_split_of_path)
    third_part_of_path = second_split_of_path[1]
    # print(third_part_of_path)

    if third_part_of_path.startswith("CT-"):
        final_part_of_path = third_part_of_path.split("-")

        if all(item in final_part_of_path for item in CT_specifications) and any(resolution in final_part_of_path[-1] for resolution in resolutions):
            print(folder_path)
        else:
            continue

    elif third_part_of_path.startswith("PT"):
        if re.search(r'\b{}\b'.format(re.escape("AC")), third_part_of_path):
            print(folder_path)
        else:
            if all(item in third_part_of_path for item in PET_specifications):
                print(folder_path)
            else:
                continue
