import os
import sys
import shutil
import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pylab as plt
import SimpleITK as sitk
from datetime import datetime

%env SITK_SHOW_COMMAND "/home/andres/Downloads/Slicer-5.4.0-linux-amd64/Slicer"
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
dicom.config.convert_wrong_length_to_UN = True

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import utils
config = utils.read_config()

source_path_wd = config['common']['paths']['source_path_wd']
source_path_bd = config['common']['paths']['source_path_bd']

master_data_wd_filename = config['metadata']['filenames']['final_selected_images_filename']
master_data_bd_filename = config['metadata']['filenames']['final_selected_images_filename']

final_selected_images_filename = config['selection']['filenames']['final_selected_images_filename']
final_selected_folders_bd = source_path_bd + '/' + final_selected_images_filename

# master_data for black disk
print("Reading master data for black disk")
master_data_bd = pd.read_excel(source_path_bd + "/" + master_data_wd_filename)
print(master_data_bd.shape)

utils.display_full(master_data_bd.head(2))

#misclassified_df = pd.read_excel(source_path_bd + '/MisclassifiedSelected_files.xlsx')
misclassified_df = pd.read_excel(source_path_bd + config["selection"]["filenames"]["misclassified_files"])
misclassified_df[['source_directory', 'patient_directory', 'PET-CT_info']] = misclassified_df['directory'].str.rsplit(pat='/', n=2, expand=True)
misclassified_patient_directory_list = misclassified_df.patient_directory.to_list()
utils.display_full(misclassified_df.head())

config_size = config['resampling']['image_size']
config_spacing = config['resampling']['voxel_spacing']
print('config_size: ', config_size)
print('config_spacing: ', config_spacing)

sample_SUV_image = sitk.ReadImage('F:/SUV.nii.gz')
print(sample_SUV_image.GetSize())
config_spacing = list(sample_SUV_image.GetSpacing())
print(config_spacing)

master_data_bd_sorted = master_data_bd.sort_values(by=["patient_directory"	,"PET-CT_info"], ascending=[True, False])
utils.display_full(master_data_bd_sorted.head(2))

master_data_bd_sorted[master_data_bd_sorted["npr"]=="npr126347730283"]
resampled_save_destination_path = config["resampling"]["path_to_save"]

exception_lst = []
resampled_SUV_CT = {'patient_directory' : [],
                    'SUV': [],
                    'CT' : [],
                    'new_size' : []}
                    #'SUV_arr': []}

for index, row in master_data_bd_sorted.iterrows():
    #print(index, row['patient_directory'], row['PET-CT_info'])
    #if index == 5:
    #   break

    #if row["npr"]=="npr126347730283" and row["scan_date"]==20170807:
    #    pass
    #else:
    #    continue
    
    #create patient directories
    npr_directories = resampled_save_destination_path + str(row['npr']) + '_SUV_CT/'
    if not os.path.exists(npr_directories):
        os.mkdir(npr_directories)
    
    #create scan date directories
    scan_date_directories = resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/'
    if not os.path.exists(scan_date_directories):
        os.mkdir(scan_date_directories)
        
    if row['modality']=='PT':

        resampled_SUV_CT['patient_directory'].append(row['patient_directory'])

        PET_img = utils.read_dicom(row['directory'])

        #save_path = resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'PT_Original'
        #utils.save_as_gz(PET_img, save_path+'.nii.gz')

        image_size = PET_img.GetSize()

        new_size = [config_size[0], config_size[1], image_size[2]]
        resampled_SUV_CT['new_size'].append(new_size)
 
        PatientWeight = float(row['patient_weight'].replace("'", "").replace(" ", ""))

        AcquisitionTime = str(row['aquisition_time'])
        AcquisitionTime = AcquisitionTime.replace("'","").strip()

        # Start Time for the Radiopharmaceutical Injection
        RadiopharmaceuticalStartTime = row['radiopharmaceutical_start_time'].split('.')[0].replace("'","").strip()

        # Half Life for Radionuclide # seconds
        RadionuclideHalfLife = row['radionuclide_half_life'].replace("'","").strip()

        # Total dose injected for Radionuclide
        RadionuclideTotalDose = row['radionuclide_total_dose'].replace("'","").strip()

        SUV, estimated, raw,spacing,origin,direction = utils.compute_suv(PET_img, PatientWeight, AcquisitionTime, RadiopharmaceuticalStartTime, RadionuclideHalfLife, RadionuclideTotalDose)
        SUV_img = sitk.GetImageFromArray(SUV)

        SUV_img.CopyInformation(PET_img)

        SUV_img = sitk.Resample(SUV_img, new_size, sitk.Transform(), sitk.sitkLinear,
                            SUV_img.GetOrigin(), config_spacing, SUV_img.GetDirection(), 0,
                           SUV_img.GetPixelID())
        
        resampled_SUV_CT['SUV'].append(row['PET-CT_info'])
        
        save_path= resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'SUV'

        utils.save_as_gz(SUV_img, save_path+'.nii.gz')
    
    if row['modality']=='CT': 

        CT_img = utils.read_dicom(row['directory'])
        
        # save_path = resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'CT_Original'
        # utils.save_as_gz(CT_img, save_path+'.nii.gz')
        SUV_img = sitk.ReadImage(resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'SUV' + '.nii.gz')
        new_size = SUV_img.GetSize()

        CT_img = sitk.Resample(CT_img, new_size, sitk.Transform(), sitk.sitkLinear,
                            SUV_img.GetOrigin(), config_spacing, SUV_img.GetDirection(), -1024,
                            CT_img.GetPixelID())
        
        CT_img.CopyInformation(SUV_img)
        
        CT_img = sitk.Clamp(CT_img, lowerBound=-1024)

        resampled_SUV_CT['CT'].append(row['PET-CT_info'])
        
        # vol_img = sitk.Clamp(vol_img,upperBound=3000)

        save_path= resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'CT'

        utils.save_as_gz(CT_img, save_path+'.nii.gz')
        
resampled_SUV_CT_df = pd.DataFrame(resampled_SUV_CT)

resampled_SUV_CT_df.shape
resampled_SUV_CT_df.to_excel(source_path_wd + '/resampled_SUV_CT_dataframe.xlsx')
