###### Logic script in order to create a dataframe that contains the exams that fit the selection rules.
###### Necessary imports and source path.
from collections import defaultdict
import pandas as pd
import os
import re
from distorted import outputDistortedImg

source_path = r"G:\ucan_lymfom/"
###### Function responsible for displaying the full information of the dataframe
def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)
###### Set of rules that affect our exam selection
### Rules for CT
CT_ignore_folders = ["bone", "lung", "lunga"]
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
###### Loading the dataset
directory_list = list()
for root, dirs, files in os.walk(source_path, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))
###### Creating a dataframe out of the dataset with the required information that are need to proceed with the filtering.
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
###### Sort the dataset according to the rules we have in order for the most desired exams to be at the top.
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
display_full(final_df['directory'])
###### Filtering the dataframe and selecting the desired exams for each patient.
# Create three lists one for CT and PET folders and one for both.
CT_selected_folders = defaultdict(list)
PET_selected_folders = defaultdict(list)
selected_exams = list()


def data_filtering(dataframe_column):
    # Looping through the dataset
    for folder_path in dataframe_column:
        
        # Splitting the name of each row in order to take only the last part 
        # that contains the CT and PET info we care about.
        first_split_of_path = folder_path.split("/")
        second_part_of_path = first_split_of_path[1]
        second_split_of_path = second_part_of_path.split("\\")
        third_part_of_path = second_split_of_path[1]
        third_part_of_path = third_part_of_path.replace("_", "-")
        
        # Check if the folder has to do with CT or PET examination.
        if third_part_of_path.startswith("CT-"):
            final_part_of_path = third_part_of_path.split("-")
            final_part_of_path = [string.lower() for string in final_part_of_path]
            resolution_str = final_part_of_path[-1][:-2]
            
            # Save the resolution to a variable for later usage.
            try:
                exam_resolution = float(resolution_str)
            except ValueError:
                continue
            resolutions_below_one = list()
            resolutions_below_one.append(exam_resolution)
            
            # Check if a CT examination has been already selected for the current patient.
            # If yes proceed with the next folder otherwise continue with the current one.
            if final_part_of_path[1] in CT_selected_folders[final_part_of_path[0]]:
                continue
            else:
                # Check if the current folder contains information that we do not find useful.
                if any(ignore in final_part_of_path for ignore in CT_ignore_folders):
                    continue
                else:
                    # The following if statements go through the set of rules that have been established
                    # in hierarchically order. Meaning that it checks if any of the folders contain information
                    # that are more important than others. The next step is to examine the resolution of the examination.
                    # Our priority is the following 3.0mm -> 2.0mm -> 1.0mm and if none of those is fulfilled 
                    # then we choose the one that is closer to one by saving the path of the file to the list of 
                    # the selected ones. Also, we save the CT selected list in order to be able to know that we 
                    # have already selected a CT examination for the current patient.  
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
                            
        # The same procedure and logic is applied for the PET folders.
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
data_filtering(final_df["directory"])
selected_exams = pd.DataFrame(selected_exams, columns=["directory"])
print("Number of images: ", selected_exams.shape[0])

selected_exams.to_csv(r"G:\ucan_lymfom/InitialSelected_exams.csv")
selected_exams.head()



distorted_lst = outputDistortedImg(selected_exams)
distorted_lst[:2]

print("Writing final distorted images directories")
with open(r'G:\ucan_lymfom\distorted_imagedirs.txt', 'w') as fp:
    for item in distorted_lst:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')
#filter distorted images from main dataframe
final_df1 = final_df[~final_df["directory"].isin(distorted_lst)].copy()
selected_exams = list() #reset
data_filtering(final_df1["directory"])
###### Generate a dataframe with the selected examinations and saving it in the form of an excel file.
# Sort the dataframe by starting from the newest examination and going to the oldest.
selected_exams = pd.DataFrame(selected_exams, columns=["directory"])
selected_exams[["source_directory", "patient_info"]] = selected_exams['directory'].str.split(pat='/', n=1, expand=True)
selected_exams[['patient_directory', 'PET-CT_info']] = selected_exams['patient_info'].str.split(pat='\\', n=1, expand=True)
selected_exams.loc[:, 'Date'] = selected_exams['patient_info'].str.split("-").str[1]
selected_exams = selected_exams.sort_values(by='Date', ascending=False)
selected_exams.reset_index(drop=True, inplace=True)
selected_exams = selected_exams.drop(columns='Date')

excel_file_location = r"G:\ucan_lymfom/Selected_exams.xlsx"
selected_exams.to_excel(excel_file_location)
