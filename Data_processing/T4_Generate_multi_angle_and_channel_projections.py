import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import os
import cv2
from tqdm import tqdm
from utils import get_2D_projections
from datetime import datetime
import traceback
from PIL import Image

# %env SITK_SHOW_COMMAND '/home/andres/Downloads/Slicer-5.4.0-linux-amd64/Slicer'

import sys

# Get the home directory
home_directory = os.path.expanduser('~')

# Append the parent directory to the Python path
sys.path.append(os.path.join(home_directory, 'VSCode', 'UCAN-PET-CT-image-data-handling-pipeline'))

from Utils import utils

config = utils.read_config()

raw_projections_path = config["projections"]["paths"]["raw_projections_path"]
resampled_SUV_CT_path = config['resampling']['path_to_save']
reshaped_projections_path = "/media/andres/T7 Shield1/UCAN_project/2D_projections/reshaped_projections/"

# Create a dataframe with the paths from the niftii files to be used later to generate the 2D projections
resampled_directory_list = []

for dirs, subdirs, files in os.walk(resampled_SUV_CT_path):
    for file in files:
        file_path = str(os.path.join(dirs, file))
        file_path = file_path.replace('\\','/')
        resampled_directory_list.append(file_path)

resampled_directory_df = pd.DataFrame(resampled_directory_list, columns=['directory'])
resampled_directory_df[['source_directory', 'patient_directory', 'scan_date', 'SUV_CT']] = resampled_directory_df['directory'].str.rsplit(pat='/', n=3, expand=True)
resampled_directory_df[['npr', 'extra']] = resampled_directory_df['patient_directory'].str.split(pat='_', n=1, expand=True)
resampled_directory_df.drop(columns=['directory','extra', 'SUV_CT'], inplace=True)
resampled_directory_df.drop_duplicates(inplace=True)

utils.display_full(resampled_directory_df.head(2))

# Generate the raw 2D projections
for index, row in resampled_directory_df.iterrows():

    CT_ptype = 'mean'
    SUV_ptype = 'max'
    angle = 90
    
    for mod in ['CT','SUV']:
        for item in ['MIP', 'bone', 'lean', 'adipose', 'air']:
            dir_path = raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/' + mod + '_' + item
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    
    CTnii_path = resampled_SUV_CT_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'CT.nii.gz'
    SUVnii_path = resampled_SUV_CT_path + str(row['npr']) + '_SUV_CT/' + str(row['scan_date']) + '/' + 'SUV.nii.gz'

    CT_img =sitk.ReadImage(CTnii_path)
    SUV_img =sitk.ReadImage(SUVnii_path)

    bone_mask, lean_mask, adipose_mask, air_mask = utils.get_proj_after_mask(CT_img)

    multiply= sitk.MultiplyImageFilter()

    CT_bone = multiply.Execute(CT_img,sitk.Cast(bone_mask,CT_img.GetPixelID()))
    CT_lean = multiply.Execute(CT_img,sitk.Cast(lean_mask,CT_img.GetPixelID()))
    CT_adipose = multiply.Execute(CT_img,sitk.Cast(adipose_mask,CT_img.GetPixelID()))
    CT_air = multiply.Execute(CT_img,sitk.Cast(air_mask,CT_img.GetPixelID()))

    utils.get_2D_projections(CT_img, 'CT', CT_ptype, angle, invert_intensity= False, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/CT_MIP/')
    utils.get_2D_projections(CT_bone, 'CT', CT_ptype, angle, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/CT_bone/')
    utils.get_2D_projections(CT_lean, 'CT', CT_ptype, angle, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/CT_lean/')
    utils.get_2D_projections(CT_adipose, 'CT', CT_ptype, angle, invert_intensity= False, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/CT_adipose/')
    utils.get_2D_projections(CT_air, 'CT', CT_ptype, angle, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/CT_air/')
    
    SUV_bone = multiply.Execute(SUV_img,sitk.Cast(bone_mask,SUV_img.GetPixelID()))
    SUV_lean = multiply.Execute(SUV_img,sitk.Cast(lean_mask,SUV_img.GetPixelID()))
    SUV_adipose = multiply.Execute(SUV_img,sitk.Cast(adipose_mask,SUV_img.GetPixelID()))
    SUV_air = multiply.Execute(SUV_img,sitk.Cast(air_mask,SUV_img.GetPixelID()))

    utils.get_2D_projections(SUV_img, 'SUV', SUV_ptype, angle, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/SUV_MIP/')
    utils.get_2D_projections(SUV_bone, 'SUV', SUV_ptype, angle, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/SUV_bone/')
    utils.get_2D_projections(SUV_lean, 'SUV', SUV_ptype, angle, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/SUV_lean/')
    utils.get_2D_projections(SUV_adipose, 'SUV', SUV_ptype, angle, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/SUV_adipose/')
    utils.get_2D_projections(SUV_air, 'SUV', SUV_ptype, angle, img_n=raw_projections_path + str(row['npr']) + '/' + str(row['scan_date']) + '/SUV_air/')

df_for_collages = pd.DataFrame(columns=["patient_ID", "scan_date", "SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air"])
for patient_ID in tqdm(sorted(os.listdir(raw_projections_path))):
    for scan_date in sorted(os.listdir(os.path.join(raw_projections_path, patient_ID))):
        for angle in ["-90.0", "0.0"]:
            SUV_MIP_path = os.path.join(raw_projections_path, patient_ID, scan_date, "SUV_MIP/" + angle + ".npy")
            SUV_bone_path = os.path.join(raw_projections_path, patient_ID, scan_date, "SUV_bone/" + angle + ".npy")
            SUV_lean_path = os.path.join(raw_projections_path, patient_ID, scan_date, "SUV_lean/" + angle + ".npy")
            SUV_adipose_path = os.path.join(raw_projections_path, patient_ID, scan_date, "SUV_adipose/" + angle + ".npy")
            SUV_air_path = os.path.join(raw_projections_path, patient_ID, scan_date, "SUV_air/" + angle + ".npy")
            df_temp = pd.DataFrame({"patient_ID": [patient_ID], "scan_date": [scan_date], "SUV_MIP": [SUV_MIP_path], "SUV_bone": [SUV_bone_path], "SUV_lean": [SUV_lean_path], "SUV_adipose": [SUV_adipose_path], "SUV_air": [SUV_air_path], "angle": [float(angle)]})
            df_for_collages = pd.concat([df_for_collages, df_temp], ignore_index=True)

# Generate a dataframe of the raw 2D projections
df_for_collages["CT_MIP"] = df_for_collages["SUV_MIP"]
df_for_collages["CT_bone"] = df_for_collages["SUV_bone"]
df_for_collages["CT_lean"] = df_for_collages["SUV_lean"]
df_for_collages["CT_adipose"] = df_for_collages["SUV_adipose"]
df_for_collages["CT_air"] = df_for_collages["SUV_air"]

df_for_collages["CT_MIP"] = df_for_collages["CT_MIP"].str.replace("SUV_MIP", "CT_MIP")
df_for_collages["CT_bone"] = df_for_collages["CT_bone"].str.replace("SUV_bone", "CT_bone")
df_for_collages["CT_lean"] = df_for_collages["CT_lean"].str.replace("SUV_lean", "CT_lean")
df_for_collages["CT_adipose"] = df_for_collages["CT_adipose"].str.replace("SUV_adipose", "CT_adipose")
df_for_collages["CT_air"] = df_for_collages["CT_air"].str.replace("SUV_air", "CT_air")


df_for_collages = df_for_collages[["patient_ID", "scan_date", "SUV_MIP", "CT_MIP", "SUV_bone", "CT_bone", "SUV_lean", "CT_lean", "SUV_adipose", "CT_adipose", "SUV_air", "CT_air", "angle"]]
df_for_collages.to_excel("/media/andres/T7 Shield1/UCAN_project/df_of_raw_projections.xlsx", index=False)

cropped_array = np.zeros((580, 256))

for row, image in df_for_collages.iterrows():

    save_path_temp = os.path.join(reshaped_projections_path, str(image["patient_ID"]), str(image["scan_date"]))
    if not os.path.exists(save_path_temp):
        os.makedirs(save_path_temp)

    SUV_MIP = np.load(image["SUV_MIP"])
    size = SUV_MIP.shape[0]

    if size <= 580:
        temp_pad_int = (580 - size)//2
        temp_pad_float = (580 - size)/2
        pad_from_top = temp_pad_int
        pad_from_bottom = temp_pad_int

        if temp_pad_int < temp_pad_float:
            pad_from_top = temp_pad_int + 1

        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_MIP" + ".npy"), result)

        SUV_bone = np.load(image["SUV_bone"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_bone" + ".npy"), result)

        SUV_lean = np.load(image["SUV_lean"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_lean" + ".npy"), result)

        SUV_adipose = np.load(image["SUV_adipose"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_adipose" + ".npy"), result)

        SUV_air = np.load(image["SUV_air"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_air" + ".npy"), result)
        
        CT_MIP = np.load(image["CT_MIP"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_MIP" + ".npy"), result)

        CT_bone = np.load(image["CT_bone"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_bone" + ".npy"), result)

        CT_lean = np.load(image["CT_lean"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_lean" + ".npy"), result)

        CT_adipose = np.load(image["CT_adipose"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_adipose" + ".npy"), result)

        CT_air = np.load(image["CT_air"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_air" + ".npy"), result)

    else:
        temp_crop_int = (size - 580)//2
        temp_crop_float = (size - 580)/2
        crop_from_top = temp_crop_int
        crop_from_bottom = crop_from_top + 580

        if temp_crop_int < temp_crop_float:
            crop_from_top = temp_crop_int + 1

        cropped_array = SUV_MIP[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_MIP" + ".npy"), cropped_array)

        SUV_bone = np.load(image["SUV_bone"])
        cropped_array = SUV_bone[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_bone" + ".npy"), cropped_array)

        SUV_lean = np.load(image["SUV_lean"])
        cropped_array = SUV_lean[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_lean" + ".npy"), cropped_array)

        SUV_adipose = np.load(image["SUV_adipose"])
        cropped_array = SUV_adipose[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_adipose" + ".npy"), cropped_array)

        SUV_air = np.load(image["SUV_air"])
        cropped_array = SUV_air[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_air" + ".npy"), cropped_array)
        
        CT_MIP = np.load(image["CT_MIP"])
        cropped_array = CT_MIP[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_MIP" + ".npy"), cropped_array)

        CT_bone = np.load(image["CT_bone"])
        cropped_array = CT_bone[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_bone" + ".npy"), cropped_array)

        CT_lean = np.load(image["CT_lean"])
        cropped_array = CT_lean[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_lean" + ".npy"), cropped_array)

        CT_adipose = np.load(image["CT_adipose"])
        cropped_array = CT_adipose[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_adipose" + ".npy"), cropped_array)

        CT_air = np.load(image["CT_air"])
        cropped_array = CT_air[crop_from_top:crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_air" + ".npy"), cropped_array)

df_for_collages.to_excel("/media/andres/T7 Shield1/UCAN_project/df_of_reshaped_projections.xlsx", index=False)
