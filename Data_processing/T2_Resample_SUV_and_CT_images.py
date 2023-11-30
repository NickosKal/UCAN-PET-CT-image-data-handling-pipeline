# import required packages
import os
import sys
import shutil
import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pylab as plt
import SimpleITK as sitk
from datetime import datetime

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
dicom.config.convert_wrong_length_to_UN = True

# parent path to read the Utils
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import main Utils package
from Utils import utils

# reading main config file
config = utils.read_config()

# reading image size and voxel spacing information
config_size = config['resampling']['image_size']
config_spacing = config['resampling']['voxel_spacing']
print('Config size: ', config_size)
print('Config spacing: ', config_spacing)

# source path for all data files
source_path_wd = config['common']['paths']['source_path_wd']
source_path_bd = config['common']['paths']['source_path_bd']

# reading filename from the config
master_data_wd_filename = config['metadata']['filenames']['final_selected_images_filename']
master_data_bd_filename = config['metadata']['filenames']['final_selected_images_filename']

final_selected_images_filename = config['selection']['filenames']['final_selected_images_filename']
final_selected_folders_bd = source_path_bd + '/' + final_selected_images_filename

# reading resampling path from the config
resampled_save_destination_path = config["resampling"]["path_to_save"]

# master_data for black disk
print("Reading master data for black disk")
master_data_bd = pd.read_excel(source_path_bd + "/" + master_data_wd_filename)
print("Master data shape: ", master_data_bd.shape)
utils.display_full(master_data_bd.head(2))

# sorting the dataframe to process PT before CT
master_data_bd_sorted = master_data_bd.sort_values(by=["patient_directory"	,"PET-CT_info"], ascending=[True, False])
utils.display_full(master_data_bd_sorted.head(2))

# create dict to store original image names
resampled_SUV_CT = {'patient_directory' : [], 'SUV': [], 'CT' : [], 'new_size' : []}

# iterating over dataframe to resampled SUV/CT and save as nifti files
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
        
    # resampling PT images    
    if row['modality']=='PT':
        # add patient directory to dict
        resampled_SUV_CT['patient_directory'].append(row['patient_directory'])
    
        # read PT dicom image
        PET_img = utils.read_dicom(row['directory'])

        #save_path = resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'PT_Original'
        #utils.save_as_gz(PET_img, save_path+'.nii.gz')

        # get PT image size
        image_size = PET_img.GetSize()

        # for new size, we include x, y dims from config and z (slice) dim from the PT image
        new_size = [config_size[0], config_size[1], image_size[2]]

        # add new size to dict
        resampled_SUV_CT['new_size'].append(new_size)

        # formatting patient weight column
        PatientWeight = float(row['patient_weight'].replace("'", "").replace(" ", ""))

        # formatting acquitition time column
        AcquisitionTime = str(row['aquisition_time'])
        AcquisitionTime = AcquisitionTime.replace("'","").strip()

        # start time for the Radiopharmaceutical Injection
        RadiopharmaceuticalStartTime = row['radiopharmaceutical_start_time'].split('.')[0].replace("'","").strip()

        # half life for Radionuclide # seconds
        RadionuclideHalfLife = row['radionuclide_half_life'].replace("'","").strip()

        # total dose injected for Radionuclide
        RadionuclideTotalDose = row['radionuclide_total_dose'].replace("'","").strip()

        # compute SUV from PT image
        SUV, estimated, raw,spacing,origin,direction = utils.compute_suv(PET_img, PatientWeight, AcquisitionTime, RadiopharmaceuticalStartTime, RadionuclideHalfLife, RadionuclideTotalDose)
        
        # get array from SUV image
        SUV_img = sitk.GetImageFromArray(SUV)

        # copy PT metadata information to SUV image
        SUV_img.CopyInformation(PET_img)

        # resample SUV to new size and new spacing
        SUV_img = sitk.Resample(SUV_img, new_size, sitk.Transform(), sitk.sitkLinear,
                            SUV_img.GetOrigin(), config_spacing, SUV_img.GetDirection(), 0,
                           SUV_img.GetPixelID())
        
        # add original PT image info to dict
        resampled_SUV_CT['SUV'].append(row['PET-CT_info'])
        
        # save resampled SUV image in nifti format
        save_path= resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'SUV'
        utils.save_as_gz(SUV_img, save_path+'.nii.gz')
    
    if row['modality']=='CT': 
        # read PT dicom image
        CT_img = utils.read_dicom(row['directory'])
        
        # save_path = resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'CT_Original'
        # utils.save_as_gz(CT_img, save_path+'.nii.gz')

        # read SUV image to get new size and metadata information
        SUV_img = sitk.ReadImage(resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'SUV' + '.nii.gz')
        new_size = SUV_img.GetSize()

        # resample CT image to new size and new spacing
        CT_img = sitk.Resample(CT_img, new_size, sitk.Transform(), sitk.sitkLinear,
                            SUV_img.GetOrigin(), config_spacing, SUV_img.GetDirection(), -1024,
                            CT_img.GetPixelID())
        
        # copy SUV metadata information to CT image
        CT_img.CopyInformation(SUV_img)
        
        # clamp lower bound intensity of CT image to -1024
        CT_img = sitk.Clamp(CT_img, lowerBound=-1024)

        # add original CT image info to dict
        resampled_SUV_CT['CT'].append(row['PET-CT_info'])
        
        # vol_img = sitk.Clamp(vol_img,upperBound=3000)

        # save resampled SUV image in nifti format
        save_path= resampled_save_destination_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'CT'
        utils.save_as_gz(CT_img, save_path+'.nii.gz')

# create resampled dataframe with original selected images info        
resampled_SUV_CT_df = pd.DataFrame(resampled_SUV_CT)

print("Shape of resampled data: ", resampled_SUV_CT_df.shape)

# save resampled dataframe
resampled_SUV_CT_df.to_excel(source_path_wd + '/resampled_SUV_CT_dataframe.xlsx')
