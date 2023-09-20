import pandas as pd
import os
import re

"""
Logic programming to select specific CT and PET images.
"""
path = "Selected_for_Sorting_test/"

# Check if bot CT and PET files exist in the patient folder
ct_folder = False
pt_folder = False

### Rules for CT
CT_specifications = ["WB", "ax", "Venfas"]
resolutions = ["1.", "2.", "3."]

### Rules for PET
PET_specifications = ["QCFX", "M.Free"]

directory_list = list()
for root, dirs, files in os.walk(path, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))

# df = pd.DataFrame(directory_list, columns=['directry'])
# # print(df.iloc[0:2, 0:])
#
# df[['sourcedir', 'patientinfo']] = df['directry'].str.split(pat='/', n=1, expand=True)
# # print(df.iloc[0:2, 0:])
#
# df[['patientdir', 'PET-CTinfo']] = df['patientinfo'].str.split(pat='\\', n=1, expand=True)
# # print(df.iloc[0:2, 0:])
#
# df[['system', 'npr', 'scandate']] = df['patientdir'].str.split(pat='_|-', n=2, expand=True)
# print(df.iloc[0:, 0:])

for folder_path in directory_list:
    first_split_of_path = folder_path.split("/")
    second_part_of_path = first_split_of_path[1]
    # print(second_part_of_path)
    second_split_of_path = second_part_of_path.split("\\")
    # print(second_split_of_path)
    # for item in second_split_of_path:
    #     print(item)
    # if second_split_of_path[1].startswith("CT-") and second_split_of_path[1].startswith("PT-"):
    #     print(f"The folder have both scans: {second_split_of_path}")
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
