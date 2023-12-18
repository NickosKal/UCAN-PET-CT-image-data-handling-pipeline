import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pylab as plt
import glob
import SimpleITK as sitk
import re
import sys

import os
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
dicom.config.convert_wrong_length_to_UN = True

parent_dir = os.path.abspath('../')
if "UCAN-PET-CT-image-data-handling-pipeline" not in parent_dir:
    parent_dir = os.path.abspath('./')

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print("parent_dir: ", parent_dir)

from Utils import utils

# reading main config file
config = utils.read_config()

system = 2 # 1 or 2
if system == 1:
    PATH = config["Source"]["paths"]["source_path_system_1"]
    DICOM_PATH = "/media/andres/T7 Shield/ucan_lymfom" #config["selection"]["filenames"]["path_to_exams_folder_system_2"]
elif system == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    PATH = config["Source"]["paths"]["source_path_system_2"]
    DICOM_PATH = "/home/ashish/T7 Shield/ucan_lymfom" #config["selection"]["filenames"]["path_to_exams_folder_system_2"]
else:
    PATH = ""
    DICOM_PATH = ""
    print("Invalid system")

#/media/ashish/T7 Shield/ucan_lymfom/ASPTCTX0001_lpr385705046400-20140313/CT-20140313-152439-6.1_UAS-WB-FDG-3D-CT_SCOUT_HEAD_IN-1250.090942mm

print("PATH: ", PATH)
print("DICOM_PATH: ", DICOM_PATH)

metadata_path = config["metadata"]["paths"]["metadata_path"]

flag_only_selected_exams = config["metadata"]["variables"]["only_selected_exams"]

if flag_only_selected_exams==True:
    #metadata_dataframe = config["metadata"]["filenames"]["metadata_all_exams_dataframe"]
    source_metadata_filename = "/Archive/Metadata/metadata.xlsx"
else:   
    #metadata_dataframe = config["metadata"]["filenames"]["metadata_selected_exams_dataframe"]
    source_metadata_filename = "/Archive/Metadata/metadata.xlsx"

source_metadata_dataframe = PATH + source_metadata_filename

linked_path = config["linked"]["paths"]["linked_path"]
finalized_dataset = config["linked"]["filenames"]["final_selected_exams_dataframe"]
finalized_dataset_exams_with_age = config["linked"]["filenames"]["regression_dataframe"]

final_selected_folders_dataframe = config["final_selected_folders_dataframe"]

selection_dataframe = PATH + final_selected_folders_dataframe
print("selection_dataframe: ", PATH + final_selected_folders_dataframe)

dataset = pd.read_excel(selection_dataframe)

if system == 1:
    dataset = dataset.replace("/media/ashish/T7 Shield/ucan_lymfom", "/media/andres/T7 Shield/ucan_lymfom", regex=True)
elif system == 2:
    dataset = dataset.replace("/media/andres/T7 Shield/ucan_lymfom", "/media/ashish/T7 Shield/ucan_lymfom", regex=True)
else:
    pass

dataset_list = dataset.directory.tolist()
print("dataset_list: ", dataset_list[:1])



directory_list = list()
for root, dirs, files in os.walk(DICOM_PATH, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))
print("directory_list: ", directory_list[:1])

remove_list = [ 'PR----BONE-PULM-mm',
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

filtered_dir_list = []
rejection_list = []

for dir in directory_list:
    dir = dir.replace('\\', '/')
    if any(item.lower() in dir.lower() for item in keep_list) and all(
            item.lower() not in dir.lower() for item in remove_list):
        #print(dir)
        filtered_dir_list.append(dir)
    else:
        rejection_list.append(dir)

print("filtered_dir_list: ", filtered_dir_list[:1])

ucan_md_dict = {'dir': [],
                'source_dir': [],
                'patient_dir': [],
                'image_dir': [],
                'dicom_img': [],
                'patient_id': [],
                'patient_age': [],
                'patient_sex': [],
                'patient_weight': [],
                'patient_size': [],
                'rows': [],
                'columns': [],
                'num_slices': [],
                'pixel_spacing': [],
                'slice_thickness': [],
                'img_pos': [],
                'img_orient': [],
                'for_uid': [],
                'att_corr': [],
                'recons_method': [],
                'image_type': [],
                'aquisition_dt': [],
                'aquisition_time': [],
                'study_desc': [],
                'series_desc': [],
                'protocol': [],
                'corr_img': [],
                'modality': [],
                'manufacturer': [],
                'manufacturer_model': [],
                'radiopharmaceutical': [],
                'radiopharmaceutical_volume': [],
                'radiopharmaceutical_start_time': [],
                'radionuclide_total_dose': [],
                'radionuclide_half_life': [],
                'radionuclide_positron_fraction': [],
                'radiopharmaceutical_start_date_time': [],
                }

count = 0

if flag_only_selected_exams==True:
    # This list is used to extract metadata for only selected exams
    final_dir_list = dataset_list           
else:
    # This list is used to extract metadata for all exams
    final_dir_list = filtered_dir_list

for dir in final_dir_list:
    #if len(os.listdir(dir))<100:
    #    emptydir_lst.append(dir)
    #    continue
    print(dir)
    ucan_md_dict['dir'].append(dir)
    ucan_md_dict['dicom_img'].append(os.listdir(dir)[0])
    
    ucan_md_dict['source_dir'].append(dir.rsplit(sep='/', maxsplit=2)[-3])
    ucan_md_dict['patient_dir'].append(dir.rsplit(sep='/', maxsplit=2)[-2])
    ucan_md_dict['image_dir'].append(dir.rsplit(sep='/', maxsplit=2)[-1])  
                                            
    #print(os.listdir(dir)[0])
    ds = dicom.dcmread(dir + '/' + str(os.listdir(dir)[0]))

    meta_info = str(ds.fix_meta_info)#.split('\n')
    
    max_range = 500
    
    #PatientID
    try:
        id = re.search(r"\b(0010, 0020)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['patient_id'].append(element)
    except:
        ucan_md_dict['patient_id'].append('')
    
    #PatientAge 
    try:
        id = re.search(r"\b(0010, 1010)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['patient_age'].append(element)
    except:
        ucan_md_dict['patient_age'].append('')

    #PatientSex
    try:
        id = re.search(r"\b(0010, 0040)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['patient_sex'].append(element)
    except:
        ucan_md_dict['patient_sex'].append('')

    #Patient Size
    try:
        id = re.search(r"\b(0010, 1020)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['patient_size'].append(element)
    except:
        ucan_md_dict['patient_size'].append('')

    #Patient Weight
    try:
        id = re.search(r"\b(0010, 1030)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['patient_weight'].append(element)
    except:
        ucan_md_dict['patient_weight'].append('')
    
    #Rows
    try:
        id = re.search(r"\b(0028, 0010)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['rows'].append(element)
    except:
        ucan_md_dict['rows'].append('')

    #Columns
    try:
        id = re.search(r"\b(0028, 0011)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['columns'].append(element)
    except:
        ucan_md_dict['columns'].append('')

    #Num of slices
    try:
        id = re.search(r"\b(07a1, 1002)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['num_slices'].append(element)
    except:
        ucan_md_dict['num_slices'].append('')

    #Pixel spacing
    try:
        id = re.search(r"\b(0028, 0030)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['pixel_spacing'].append(element)
    except:
        ucan_md_dict['pixel_spacing'].append('')

    #Slice thickness
    try:
        id = re.search(r"\b(0018, 0050)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['slice_thickness'].append(element)
    except:
        ucan_md_dict['slice_thickness'].append('')

    #Image positioning
    try:
        id = re.search(r"\b(0020, 0032)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['img_pos'].append(element)
    except:
        ucan_md_dict['img_pos'].append('')

    #Image orientation
    try:
        id = re.search(r"\b(0020, 0037)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['img_orient'].append(element)
    except:
        ucan_md_dict['img_orient'].append('')

    #Frame of reference UID
    try:
        id = re.search(r"\b(0020, 0052)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['for_uid'].append(element)
    except:
        ucan_md_dict['for_uid'].append('')

    #Attenuation Correction Method
    try:
        id = re.search(r"\b(0054, 1101)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['att_corr'].append(element)
    except:
        ucan_md_dict['att_corr'].append('')

    #Reconstruction Method
    try:
        id = re.search(r"\b(0054, 1103)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['recons_method'].append(element)
    except:
        ucan_md_dict['recons_method'].append('')

    #Image Type
    try:
        id = re.search(r"\b(0008, 0008)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['image_type'].append(element)
    except:
        ucan_md_dict['image_type'].append('')

    #Aquisition Date
    try:
        id = re.search(r"\b(0008, 0022)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['aquisition_dt'].append(element)
    except:
        ucan_md_dict['aquisition_dt'].append('')
    
    #Aquisition Time
    try:
        id = re.search(r"\b(0008, 0032)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['aquisition_time'].append(element)
    except:
        ucan_md_dict['aquisition_time'].append('')

    #Study Description
    try:
        id = re.search(r"\b(0008, 1030)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['study_desc'].append(element)
    except:
        ucan_md_dict['study_desc'].append('')

    #Series Description
    try:
        id = re.search(r"\b(0008, 103e)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['series_desc'].append(element)
    except:
        ucan_md_dict['series_desc'].append('')

    #Protocol Name 
    try:
        id = re.search(r"\b(0018, 1030)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['protocol'].append(element)
    except:
        ucan_md_dict['protocol'].append('')

    #Corrected Image
    try:
        id = re.search(r"\b(0028, 0051)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['corr_img'].append(element)
    except:
        ucan_md_dict['corr_img'].append('')

    #Modality
    try:
        id = re.search(r"\b(0008, 0060)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['modality'].append(element)
    except:
        ucan_md_dict['modality'].append('')

    #Manufacturer
    try:
        id = re.search(r"\b(0008, 0070)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['manufacturer'].append(element)
    except:
        ucan_md_dict['manufacturer'].append('')

    #Manufacturer_model
    try:
        id = re.search(r"\b(0008, 1090)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['manufacturer_model'].append(element)
    except:
        ucan_md_dict['manufacturer_model'].append('')
    # Radiopharmaceutical
    try:
        id = re.search(r"\b(0018, 0031)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['radiopharmaceutical'].append(element)
    except:
        ucan_md_dict['radiopharmaceutical'].append('')
     # Radiopharmaceutical Volume
    try:
        id = re.search(r"\b(0018, 1071)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['radiopharmaceutical_volume'].append(element)
    except:
        ucan_md_dict['radiopharmaceutical_volume'].append('')
     # Radiopharmaceutical Start Time
    try:
        id = re.search(r"\b(0018, 1072)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['radiopharmaceutical_start_time'].append(element)
    except:
        ucan_md_dict['radiopharmaceutical_start_time'].append('')
     # Radionuclide Total Dose
    try:
        id = re.search(r"\b(0018, 1074)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['radionuclide_total_dose'].append(element)
    except:
        ucan_md_dict['radionuclide_total_dose'].append('')
     # Radionuclide Half Life
    try:
        id = re.search(r"\b(0018, 1075)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['radionuclide_half_life'].append(element)
    except:
        ucan_md_dict['radionuclide_half_life'].append('')
     # Radionuclide Positron Fraction
    try:
        id = re.search(r"\b(0018, 1076)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['radionuclide_positron_fraction'].append(element)
    except:
        ucan_md_dict['radionuclide_positron_fraction'].append('')
     # Radiopharmaceutical Start Date Time
    try:
        id = re.search(r"\b(0018, 1078)\b", meta_info).start()
        element = meta_info[id:id+max_range].split('\n')[0].split(':')[1]
        ucan_md_dict['radiopharmaceutical_start_date_time'].append(element)
    except:
        ucan_md_dict['radiopharmaceutical_start_date_time'].append('')

    if count>1:
        break
    count += 1

print("ucan_md_dict:", ucan_md_dict)

ucan_md = pd.DataFrame(ucan_md_dict)
print(ucan_md.shape)
#(8325, 29)
#(8275, 29)


print("Show column data types: \n", ucan_md.dtypes)
print("Cleaning columns and correcting data types in metadata")
ucan_md['imgsz_x'] = np.int64(ucan_md['rows'].str.replace("'","").replace(" ",""))
ucan_md['imgsz_y'] = np.int64(ucan_md['columns'])
ucan_md['num_slices'] = np.int64(ucan_md['num_slices'])
ucan_md['voxsz_x'] = ucan_md['pixel_spacing'].apply(lambda x: (np.float64(x.replace(" ","").replace("[","").replace("]","").split(',')[0])))
ucan_md['voxsz_y'] = ucan_md['pixel_spacing'].apply(lambda x: (np.float64(x.replace(" ","").replace("[","").replace("]","").split(',')[1])))
ucan_md['slice_thickness'] = np.float64(ucan_md['slice_thickness'].str.replace("'","").replace(" ",""))

print("Creating new voxel_size and image_size columns by combining multiple columns in metadata")
ucan_md['image_size'] = ucan_md.apply(lambda x: (x.imgsz_x, x.imgsz_y, x.num_slices), axis=1)
ucan_md['voxel_size'] = ucan_md.apply(lambda x: (x.voxsz_x, x.voxsz_y, x.slice_thickness), axis=1)

#Reordering columns
print("Reordering columns in desired order in metadata")
new_col_lst = ['dir', 'source_dir', 'patient_dir', 'image_dir', 'dicom_img',
       'patient_id', 'patient_age', 'patient_sex', 'patient_weight',
       'patient_size', 'rows', 'columns', 'imgsz_x', 'imgsz_y', 'num_slices', 
       'voxsz_x', 'voxsz_y', 'slice_thickness',  'pixel_spacing',
       'image_size', 'voxel_size', 'img_pos', 'img_orient', 'for_uid', 'att_corr',
       'recons_method', 'image_type', 'aquisition_dt', 'aquisition_time', 'study_desc',
       'series_desc', 'protocol', 'corr_img', 'modality', 'manufacturer',
       'manufacturer_model', 'radiopharmaceutical', 'radiopharmaceutical_volume',
       'radiopharmaceutical_start_time', 'radionuclide_total_dose', 
       'radionuclide_half_life', 'radionuclide_positron_fraction', 
       'radiopharmaceutical_start_date_time'
       ]

ucan_md = ucan_md[new_col_lst]

# Checking total number of CT and PET folders in metadata
print("Total CT folders: ", ucan_md[ucan_md['image_dir'].str.contains('CT')].shape[0])
print("Total PT folders: ", ucan_md[ucan_md['image_dir'].str.contains('PT')].shape[0])

# Saving source metadata dataframe
#ucan_md.to_excel(source_metadata_dataframe, index=False)

# Analyzing and saving data in different number of slices group
if flag_only_selected_exams==True:
    print("\nAnalyzing and saving results in different number of slices group from metadata extracted for only selected exams:")
    # exams with less than 100 slices
    selected_exams_with_less_than_100_slices = ucan_md[np.int64(ucan_md['num_slices'])<=100].groupby(['modality', 'slice_thickness', 'num_slices']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #selected_exams_with_less_than_100_slices.to_excel(os.path.join(metadata_path, 'selected_exams_with_less_than_100_slices.xlsx'))
    print("selected_exams_with_less_than_100_slices shape: ", selected_exams_with_less_than_100_slices.shape)

    # voxel distribution of exams with less than 100 slices
    selected_exams_with_less_than_100_slices_voxeldist = selected_exams_with_less_than_100_slices.groupby(['modality', 'voxel_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #selected_exams_with_less_than_100_slices_voxeldist.to_excel(os.path.join(metadata_path, 'selected_exams_with_less_than_100_slices_voxeldist.xlsx'))

    # image size distribution of exams with less than 100 slices
    selected_exams_with_less_than_100_slices_voxeldist_imgszdist = selected_exams_with_less_than_100_slices.groupby(['modality', 'image_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #selected_exams_with_less_than_100_slices_voxeldist_imgszdist.to_excel(os.path.join(metadata_path, 'selected_exams_with_less_than_100_slices_imgszdist.xlsx'))

    # exams with greater than and equal 200 slices
    selected_exams_with_greater_than_200_slices = ucan_md[ucan_md['num_slices']>=200]
    #selected_exams_with_greater_than_200_slices.to_excel(os.path.join(metadata_path, 'selected_exams_with_greater_than_200_slices.xlsx'))
    print("selected_exams_with_greater_than_200_slices shape: ", selected_exams_with_greater_than_200_slices.shape)

    # voxel distribution of exams with greater than and equal 200 slices
    selected_exams_with_greater_than_200_slices_voxeldist = selected_exams_with_greater_than_200_slices.groupby(['modality', 'voxel_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #selected_exams_with_greater_than_200_slices_voxeldist.to_excel(os.path.join(metadata_path, 'selected_exams_with_greater_than_200_slices_voxeldist.xlsx'))

    # image size distribution of exams with greater than and equal 200 slices
    selected_exams_with_greater_than_200_slices_imgszdist = selected_exams_with_greater_than_200_slices.groupby(['modality', 'image_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #selected_exams_with_greater_than_200_slices_imgszdist.to_excel(os.path.join(metadata_path, 'selected_exams_with_greater_than_200_slices_imgszdist.xlsx'))

    # exams with greater than and equal 200 slices
    selected_exams_with_less_than_200_slices = ucan_md[ucan_md['num_slices']<200]
    #selected_exams_with_less_than_200_slices.to_excel(os.path.join(metadata_path, "selected_exams_with_less_than_200_slices.xlsx"))
    print("selected_exams_with_less_than_200_slices shape: ", selected_exams_with_less_than_200_slices.shape)

    # voxel distribution of exams with greater than and equal 200 slices
    selected_exams_with_less_than_200_slices_voxeldist = selected_exams_with_less_than_200_slices.groupby(['modality', 'voxel_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #selected_exams_with_less_than_200_slices_voxeldist.to_excel(os.path.join(metadata_path, 'selected_exams_with_less_than_200_slices_voxeldist.xlsx'))

    # image size distribution of exams with greater than and equal 200 slices
    selected_exams_with_less_than_200_slices_imgszdist = selected_exams_with_less_than_200_slices.groupby(['modality', 'image_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #selected_exams_with_less_than_200_slices_imgszdist.to_excel(os.path.join(metadata_path, 'selected_exams_with_less_than_200_slices_imgszdist.xlsx'))
else:
    print("\nAnalyzing and saving results in different number of slices group from metadata extracted for all exams:")
    # exams with less than 100 slices
    all_exams_with_less_than_100_slices = ucan_md[np.int64(ucan_md['num_slices'])<=100].groupby(['modality', 'slice_thickness', 'num_slices']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #all_exams_with_less_than_100_slices.to_excel(os.path.join(metadata_path, 'all_exams_with_less_than_100_slices.xlsx'))
    print("all_exams_with_less_than_100_slices shape: ", all_exams_with_less_than_100_slices.shape)

    # voxel distribution of exams with less than 100 slices
    all_exams_with_less_than_100_slices_voxeldist = all_exams_with_less_than_100_slices.groupby(['modality', 'voxel_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #all_exams_with_less_than_100_slices_voxeldist.to_excel(os.path.join(metadata_path, 'all_exams_with_less_than_100_slices_voxeldist.xlsx'))

    # image size distribution of exams with less than 100 slices
    all_exams_with_less_than_100_slices_voxeldist_imgszdist = all_exams_with_less_than_100_slices.groupby(['modality', 'image_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #all_exams_with_less_than_100_slices_voxeldist_imgszdist.to_excel(os.path.join(metadata_path, 'all_exams_with_less_than_100_slices_voxeldist_imgszdist.xlsx'))

    # exams with greater than and equal 200 slices
    all_exams_with_greater_than_200_slices = ucan_md[ucan_md['num_slices']>=200]
    #all_exams_with_greater_than_200_slices.to_excel(os.path.join(metadata_path, 'all_exams_with_greater_than_200_slices.xlsx'))
    print("all_exams_with_greater_than_200_slices shape: ", all_exams_with_greater_than_200_slices.shape)

    # voxel distribution of exams with greater than and equal 200 slices
    all_exams_with_greater_than_200_slices_voxeldist = all_exams_with_greater_than_200_slices.groupby(['modality', 'voxel_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #all_exams_with_greater_than_200_slices_voxeldist.to_excel(os.path.join(metadata_path, 'all_exams_with_greater_than_200_slices_voxeldist.xlsx'))

    # image size distribution of exams with greater than and equal 200 slices
    all_exams_with_greater_than_200_slices_imgszdist = all_exams_with_greater_than_200_slices.groupby(['modality', 'image_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #all_exams_with_greater_than_200_slices_imgszdist.to_excel(os.path.join(metadata_path, 'all_exams_with_greater_than_200_slices_imgszdist.xlsx'))

    # exams with greater than and equal 200 slices
    all_exams_with_less_than_200_slices = ucan_md[ucan_md['num_slices']<200]
    #all_exams_with_less_than_200_slices.to_excel(os.path.join(metadata_path, "all_exams_with_less_than_200_slices.xlsx"))
    print("all_exams_with_less_than_200_slices shape: ", all_exams_with_less_than_200_slices.shape)

    # voxel distribution of exams with greater than and equal 200 slices
    all_exams_with_less_than_200_slices_voxeldist = all_exams_with_less_than_200_slices.groupby(['modality', 'voxel_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #all_exams_with_less_than_200_slices_voxeldist.to_excel(os.path.join(metadata_path, 'all_exams_with_less_than_200_slices_voxeldist.xlsx'))

    # image size distribution of exams with greater than and equal 200 slices
    all_exams_with_less_than_200_slices_imgszdist = all_exams_with_less_than_200_slices.groupby(['modality', 'image_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
    #all_exams_with_less_than_200_slices_imgszdist.to_excel(os.path.join(metadata_path, 'all_exams_with_less_than_200_slices_imgszdist.xlsx'))


# Link Metadata with Selected Exams
metadata = pd.read_excel(source_metadata_dataframe)
metadata['dir'] = metadata['dir'].str.replace('\\','/')
print("Metadata shape: ", metadata.shape)

selected_imgs = pd.read_excel(selection_dataframe)
print("Selected exams shape: ", selected_imgs.shape)

print("Number of exam in the selected_imgs dataframe", selected_imgs.patient_directory.nunique())
print("Number of patients in the selected_imgs dataframe", selected_imgs.npr.nunique())

print("Merging metadata with selected exams data")
master_data = pd.merge(selected_imgs, metadata, how="inner", left_on=['patient_directory', 'PET-CT_info'], right_on=['patient_dir', 'image_dir'], sort=True, suffixes=("_x", "_y"))

col_drop_lst = ['dir', 'source_dir', 'patient_dir',	'image_dir', 'patient_sex', 'rows',	'columns', 'pixel_spacing']

print("Dropping unnecessary column from merged data")
master_data = master_data.drop(columns=col_drop_lst)

print("Cleaning columns in merged data")
master_data['patient_id'] = master_data['patient_id'].apply(lambda x: x.split('_')[1].replace("'",""))
master_data['patient_age'] = master_data['patient_age'].apply(lambda x: x.replace("'","").replace("Y","").replace(" ",""))#.astype('Int64')
master_data['modality'] = master_data['modality'].apply(lambda x: x.strip().replace("'","").strip())

#master_data.to_excel(finalized_dataset)

# Check no of patients having age in the metadata
print("Master data shape: ", master_data.shape)
print("Total number of patients: ", master_data.npr.nunique())
print("Total number of exams: ", master_data.patient_directory.nunique())
print("Unique patient age in the data: ", master_data.patient_age.unique())

# Create dataset with only those patients who have age info in the metadata
master_data_having_age = master_data[master_data.patient_age>0].copy()
print("Master data shape: ", master_data_having_age.shape)
print("Total number of patients: ", master_data_having_age.npr.nunique())
print("Total number of exams: ", master_data_having_age.patient_directory.nunique())

#master_data_having_age.to_excel(finalized_dataset_exams_with_age)

print("\nSummary for CT images:")
print("Mode of x image dims: ", master_data[master_data['modality']=='CT']['imgsz_x'].mode()[0])
print("Mode of y image dims: ", master_data[master_data['modality']=='CT']['imgsz_y'].mode()[0])
print("Mode of z image dims: ", master_data[master_data['modality']=='CT']['num_slices'].mode()[0])
print("Mode of x voxel dims: ", master_data[master_data['modality']=='CT']['voxsz_x'].mode()[0])
print("Mode of y voxel dims: ", master_data[master_data['modality']=='CT']['voxsz_y'].mode()[0])
print("Mode of z voxel dims: ", master_data[master_data['modality']=='CT']['slice_thickness'].mode()[0])
utils.display_full(master_data[master_data['modality']=='CT'][['imgsz_x','imgsz_y','num_slices','voxsz_x','voxsz_y','slice_thickness']].describe())

print("\nSummary for PET images:")
print("Mode of x image dims: ", master_data[master_data['modality']=='PT']['imgsz_x'].mode()[0])
print("Mode of y image dims: ", master_data[master_data['modality']=='PT']['imgsz_y'].mode()[0])
print("Mode of z image dims: ", master_data[master_data['modality']=='PT']['num_slices'].mode()[0])
print("Mode of x voxel dims: ", master_data[master_data['modality']=='PT']['voxsz_x'].mode()[0])
print("Mode of y voxel dims: ", master_data[master_data['modality']=='PT']['voxsz_y'].mode()[0])
print("Mode of z voxel dims: ", master_data[master_data['modality']=='PT']['slice_thickness'].mode()[0])
utils.display_full(master_data[master_data['modality']=='PT'][['imgsz_x','imgsz_y','num_slices','voxsz_x','voxsz_y','slice_thickness']].describe())