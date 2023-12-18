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

#metadata_dataframe = config["metadata_dataframe"]
metadata_dataframe = "/Archive/Metadata/metadata.xlsx"

final_selected_folders_dataframe = config["final_selected_folders_dataframe"]

destination_path = PATH + metadata_dataframe

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

findir_lst = []
rejection_lst = []

for dir in directory_list:
    dir = dir.replace('\\', '/')
    if any(item.lower() in dir.lower() for item in keep_list) and all(
            item.lower() not in dir.lower() for item in remove_list):
        #print(dir)
        findir_lst.append(dir)
    else:
        rejection_lst.append(dir)

print("findir_lst: ", findir_lst[:1])

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
for dir in dataset_list:
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
#print(ucan_md.head(2))

ucan_md.dtypes
ucan_md['imgsz_x'] = np.int64(ucan_md['rows'].str.replace("'","").replace(" ",""))
ucan_md['imgsz_y'] = np.int64(ucan_md['columns'])
ucan_md['num_slices'] = np.int64(ucan_md['num_slices'])
ucan_md['voxsz_x'] = ucan_md['pixel_spacing'].apply(lambda x: (np.float64(x.replace(" ","").replace("[","").replace("]","").split(',')[0])))
ucan_md['voxsz_y'] = ucan_md['pixel_spacing'].apply(lambda x: (np.float64(x.replace(" ","").replace("[","").replace("]","").split(',')[1])))
ucan_md['slice_thickness'] = np.float64(ucan_md['slice_thickness'].str.replace("'","").replace(" ",""))

ucan_md['image_size'] = ucan_md.apply(lambda x: (x.imgsz_x, x.imgsz_y, x.num_slices), axis=1)
ucan_md['voxel_size'] = ucan_md.apply(lambda x: (x.voxsz_x, x.voxsz_y, x.slice_thickness), axis=1)
#ucan_md.head(2)

#print(ucan_md.head(2))

#order columns
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

print("Toal CT folders: ", ucan_md[ucan_md['image_dir'].str.contains('CT')].shape[0])
print("Toal PT folders: ", ucan_md[ucan_md['image_dir'].str.contains('PT')].shape[0])

ucan_md.to_excel(destination_path)

#ucan_md[ucan_md['num_slices']<=100].to_excel(os.path.join(source_path, 'exams_with_less_than_100_slices.xlsx'))

lessthan100slices_summ = ucan_md[np.int64(ucan_md['num_slices'])<=100].groupby(['modality', 'slice_thickness', 'num_slices']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
#lessthan100slices_summ.to_excel(os.path.join(source_path, 'exams_with_less_than_100_slices_final.xlsx'))

print(lessthan100slices_summ.shape)

ucan_md_voxeldist = ucan_md.groupby(['modality', 'voxel_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
#ucan_md_voxeldist.to_excel(os.path.join(source_path, 'ucan_exams_md_voxeldist.xlsx'))
ucan_md_voxeldist.head(10)

ucan_md_imgszdist = ucan_md.groupby(['modality', 'image_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
#ucan_md_imgszdist.to_excel(os.path.join(source_path, 'ucan_exams_md_imgszdist.xlsx'))
ucan_md_imgszdist.head(10)

ucan_md_gt200slices = ucan_md[ucan_md['num_slices']>=200]
#ucan_md_gt200slices.to_excel(os.path.join(source_path, 'ucan_exams_with_more_than_200_slices.xlsx'))
print(ucan_md_gt200slices.shape)
ucan_md_gt200slices.head(2)

ucan_md_gt200slices_voxeldist = ucan_md_gt200slices.groupby(['modality', 'voxel_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
#ucan_md_gt200slices_voxeldist.to_excel(os.path.join(source_path, 'ucan_exams_with_more_than_200_slices_voxeldist.xlsx'))
ucan_md_gt200slices_voxeldist.head(10)

ucan_md_gt200slices_imgszdist = ucan_md_gt200slices.groupby(['modality', 'image_size']).agg({'dicom_img':'count'}).sort_values('dicom_img' ,ascending=False).reset_index()
#ucan_md_gt200slices_imgszdist.to_excel(os.path.join(source_path, 'ucan_exams_with_more_than_200_slices_imagesizedist.xlsx'))
ucan_md_gt200slices_imgszdist.head(10)

Lessthan_200Slices = ucan_md[ucan_md['num_slices']<200]
#Lessthan_200Slices.to_excel(os.path.join(source_path, "less_than_200_slices.xlsx"))
print(Lessthan_200Slices.shape)
Lessthan_200Slices.head(2)


# Match Metadata with Selected Files
metadata = pd.read_excel(destination_path)
# metadata['dir'] = metadata['dir'].str.replace('/','\\')
print(metadata.shape)
metadata.head(2)

selected_imgs = pd.read_excel(selection_dataframe)
print(selected_imgs.shape)
selected_imgs.head(2)

print("Number of exam in the selected_imgs dataframe", selected_imgs.patient_directory.nunique())


master_data = pd.merge(selected_imgs, metadata, how="inner", left_on=['patient_directory', 'PET-CT_info'], right_on=['patient_dir', 'image_dir'], sort=True, suffixes=("_x", "_y"))

col_drop_lst = ['dir', 'source_dir', 'patient_dir',	'image_dir', 'patient_sex', 'rows',	'columns', 'pixel_spacing']

master_data = master_data.drop(columns=col_drop_lst)

master_data['patient_id'] = master_data['patient_id'].apply(lambda x: x.split('_')[1].replace("'",""))
master_data['patient_age'] = master_data['patient_age'].apply(lambda x: x.replace("'","").replace("Y","").replace(" ",""))#.astype('Int64')
master_data['modality'] = master_data['modality'].apply(lambda x: x.strip().replace("'","").strip())

#master_data.to_excel(os.path.join(source_path, "Excel_files/06_11_2023/Finalized_dataset.xlsx"))
#print(master_data.head())

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

master_data_having_age.to_excel(os.path.join(source_path, "Excel_files/06_11_2023/Finalized_dataset_1805_exams_with_Age.xlsx"))


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