import pandas as pd
import os
"""
Logic programming to select specific CT and PET images.
The following acronyms stand for:
fp -> folder_path
fsop -> first_split_of_path
spop -> second_part_of_path
ssop -> second_split_of_path
tpop -> third_part_of_path
fpop -> final_part_of_path
"""
path = "Selected_for_Sorting_test/"

### Rules for CT
CT_specifications = ["WB", "ax", "Venfas"]
resolutions = ["1.", "2.", "3."]

### Rules for PET
PET_specifications = ["QCFX", ]

directory_list = list()
for root, dirs, files in os.walk(path, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))

# df = pd.DataFrame(directory_list, columns=['directry'])
# print(df.head(2))
#
# df[['sourcedir', 'patientinfo']] = df['directry'].str.split('/', 1, expand=True)
# print(df.head(2))
#
# df[['patientdir', 'PET-CTinfo']] = df['patientinfo'].str.split('\\', 1, expand=True)
# df.head(2)
#
# df[['system', 'npr', 'scandate']] = df['patientdir'].str.split('_|-', 2, expand=True)
# df.head(2)

for fp in directory_list:
    fsop = fp.split("/")
    spop = fsop[1]
    ssop = spop.split("\\")
    tpop = ssop[1]

    if tpop.startswith("CT-"):
        fpop = tpop.split("-")

        if all(item in fpop for item in CT_specifications) and any(resolution in fpop[-1] for resolution in resolutions):
            print(fp)

    elif tpop.startswith("PT"):
        if "QCFX" in tpop:
            print(fp)

