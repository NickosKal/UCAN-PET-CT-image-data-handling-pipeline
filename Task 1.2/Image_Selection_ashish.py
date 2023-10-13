###### Logic script in order to create a dataframe that contains the exams that fit the selection rules.
###### Necessary imports and source path.
from collections import defaultdict
import os
import re
import sys
import time
from datetime import datetime
import psutil
import numpy  as np
import pandas as pd
import concurrent.futures
import pydicom as dicom
import SimpleITK as sitk
#from distorted import outputDistortedImg
import random
random.seed(10)

dicom.config.convert_wrong_length_to_UN = True

source_path = r"G:\ucan_lymfom/"
###### Function responsible for displaying the full information of the dataframe
def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)

def checkDistortedImg(vol_img,ptype='mean',angle=90):
    projection = {'sum': sitk.SumProjection,
                'mean':  sitk.MeanProjection,
                'std': sitk.StandardDeviationProjection,
                'min': sitk.MinimumProjection,
                'max': sitk.MaximumProjection}
    paxis = 0

    rotation_axis = [0,0,1]
    rotation_angles = np.linspace(-1/2*np.pi, 1/2*np.pi, int(180.0/angle)) #15.0 degree 

    rotation_center = vol_img.TransformContinuousIndexToPhysicalPoint([(index-1)/2.0 for index in vol_img.GetSize()])

    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_center)

    #Compute bounding box of rotating volume and the resampling grid structure
    image_indexes = list(zip([0,0,0], [sz-1 for sz in vol_img.GetSize()]))
    image_bounds = []
    for i in image_indexes[0]:
        for j in image_indexes[1]:
            for k in image_indexes[2]:
                image_bounds.append(vol_img.TransformIndexToPhysicalPoint([i,j,k]))

    all_points = []
    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle)    
        all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])
        
    all_points = np.array(all_points)
    min_bounds = all_points.min(0)
    max_bounds = all_points.max(0)

    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]

    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle) 
        resampled_image = sitk.Resample(image1=vol_img,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkLinear,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection = [1,0,0,0,1,0,0,0,1],
                                        defaultPixelValue =  -20, #HU unit for air in CT, possibly set to 0 in other cases
                                        outputPixelType = vol_img.GetPixelID())
        proj_image = projection[ptype](resampled_image, paxis)
        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0
        sitk.Extract(proj_image, extract_size)

def outputDistortedImg(df):
    pid  = os.getpid()
    ppid = os.getppid()
    start = time.time()
    print("PPID %s->%s Started on %s"%(ppid, pid, str(datetime.now())))
    
    exception_lst = []

    for _, row in df.iterrows():
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(row['directory'])
        reader.SetFileNames(dicom_names)
        vol_img = reader.Execute()
        try:
            checkDistortedImg(vol_img)
        except:
            exception_lst.append(row['directory'])

    end = time.time()

    print("PPID %s Completed in %s"%(os.getpid(), round((end-start)/60,2)))

    return exception_lst


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

if __name__ == '__main__':

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


    print("Running the data filtering - initial run")
    data_filtering(final_df["directory"])
    selected_exams = pd.DataFrame(selected_exams, columns=["directory"])
    print("Number of images: ", selected_exams.shape[0])

    logical    = False
    distorted_lst = []
    num_procs  = psutil.cpu_count(logical=logical)
    if len(sys.argv) > 1:
        num_procs = int(sys.argv[1])
    print("Number of processes available: ", num_procs)

    print("Splitting dataframe into ", num_procs-1, " dataframes")
    splitted_df = np.array_split(selected_exams, num_procs-1)
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
        results = [ executor.submit(outputDistortedImg,df=df) for df in splitted_df ]
        for result in concurrent.futures.as_completed(results):
            try:
                distorted_lst.append(result.result())
            except Exception as ex:
                print(str(ex))
                pass
    end = time.time()

    print("Writing final distorted images lst")
    with open(r'F:\U-CAN-Lymfom_A\Selected_for_UCAN_project\DistortedImageTest\distorted_lst.txt', 'w') as fp:
        for item in distorted_lst:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

    print("-------------------------------------------")
    print("PPID %s Completed in %s"%(os.getpid(), round(end-start,2)))



    #distorted_lst = outputDistortedImg(selected_exams)
    #distorted_lst[:2]

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
