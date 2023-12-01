import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import scipy.ndimage
import cv2
import os

import sys

# Get the home directory
home_directory = os.path.expanduser('~')

# Append the parent directory to the Python path
sys.path.append(os.path.join(home_directory, 'VSCode', 'UCAN-PET-CT-image-data-handling-pipeline'))

from Utils import utils

def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)

raw_collages_path = "/media/andres/T7 Shield1/UCAN_project/collages/raw_collages"
df_of_raw_projections = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/df_of_raw_projections.xlsx")
df_of_raw_projections.head()

df_of_raw_projections["scan_date"] = df_of_raw_projections["scan_date"].astype(str)
df_of_raw_projections["unique_pat_ID_scan_date"] = df_of_raw_projections["patient_ID"] + "_" + df_of_raw_projections["scan_date"]
unique_patient = np.unique(df_of_raw_projections["unique_pat_ID_scan_date"])

# Generate the collages
for scan_date in tqdm(unique_patient):
    temp = df_of_raw_projections[df_of_raw_projections["unique_pat_ID_scan_date"] == scan_date]
    save_path = os.path.join(raw_collages_path, str(temp["patient_ID"].iloc[0]), str(temp["scan_date"].iloc[0]))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    SUV_MIP_list = []
    SUV_bone_list = []
    SUV_lean_list = []
    SUV_adipose_list = []
    SUV_air_list = []

    CT_MIP_list = []
    CT_bone_list = []
    CT_lean_list = []
    CT_adipose_list = []
    CT_air_list = []

    for idx, row in temp.iterrows():
        SUV_MIP_list.append(np.load(row["SUV_MIP"]))
        SUV_bone_list.append(np.load(row["SUV_bone"]))
        SUV_lean_list.append(np.load(row["SUV_lean"]))
        SUV_adipose_list.append(np.load(row["SUV_adipose"]))
        SUV_air_list.append(np.load(row["SUV_air"]))
        
        CT_MIP_list.append(np.load(row["CT_MIP"]))
        CT_bone_list.append(np.load(row["CT_bone"]))
        CT_lean_list.append(np.load(row["CT_lean"]))
        CT_adipose_list.append(np.load(row["CT_adipose"]))
        CT_air_list.append(np.load(row["CT_air"]))
    
    SUV_MIP = np.concatenate((SUV_MIP_list[0], SUV_MIP_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_MIP.npy"), SUV_MIP)

    SUV_bone = np.concatenate((SUV_bone_list[0], SUV_bone_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_bone.npy"), SUV_bone)

    SUV_lean = np.concatenate((SUV_lean_list[0], SUV_lean_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_lean.npy"), SUV_lean)

    SUV_adipose = np.concatenate((SUV_adipose_list[0], SUV_adipose_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_adipose.npy"), SUV_adipose)

    SUV_air = np.concatenate((SUV_air_list[0], SUV_air_list[1]), axis=1)
    np.save(os.path.join(save_path, "SUV_air.npy"), SUV_air)

    CT_MIP = np.concatenate((CT_MIP_list[0], CT_MIP_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_MIP.npy"), CT_MIP)

    CT_bone = np.concatenate((CT_bone_list[0], CT_bone_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_bone.npy"), CT_bone)

    CT_lean = np.concatenate((CT_lean_list[0], CT_lean_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_lean.npy"), CT_lean)

    CT_adipose = np.concatenate((CT_adipose_list[0], CT_adipose_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_adipose.npy"), CT_adipose)

    CT_air = np.concatenate((CT_air_list[0], CT_air_list[1]), axis=1)
    np.save(os.path.join(save_path, "CT_air.npy"), CT_air)

# Create a dataframe with the paths of the collages.
df_of_collages = pd.DataFrame(columns=["patient_ID", "scan_date", "SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air"])
for patient_ID in tqdm(sorted(os.listdir(raw_collages_path))):
    for scan_date in sorted(os.listdir(os.path.join(raw_collages_path, patient_ID))):
        for angle in ["-90.0", "0.0"]:
            SUV_MIP_path = os.path.join(raw_collages_path, patient_ID, scan_date, "SUV_MIP" + ".npy")
            SUV_bone_path = os.path.join(raw_collages_path, patient_ID, scan_date, "SUV_bone" + ".npy")
            SUV_lean_path = os.path.join(raw_collages_path, patient_ID, scan_date, "SUV_lean" + ".npy")
            SUV_adipose_path = os.path.join(raw_collages_path, patient_ID, scan_date, "SUV_adipose" + ".npy")
            SUV_air_path = os.path.join(raw_collages_path, patient_ID, scan_date, "SUV_air" + ".npy")
            df_temp = pd.DataFrame({"patient_ID": [patient_ID], "scan_date": [scan_date], "SUV_MIP": [SUV_MIP_path], "SUV_bone": [SUV_bone_path], "SUV_lean": [SUV_lean_path], "SUV_adipose": [SUV_adipose_path], "SUV_air": [SUV_air_path]})
            df_of_collages = pd.concat([df_of_collages, df_temp], ignore_index=True)

df_of_collages["CT_MIP"] = df_of_collages["SUV_MIP"]
df_of_collages["CT_bone"] = df_of_collages["SUV_bone"]
df_of_collages["CT_lean"] = df_of_collages["SUV_lean"]
df_of_collages["CT_adipose"] = df_of_collages["SUV_adipose"]
df_of_collages["CT_air"] = df_of_collages["SUV_air"]

df_of_collages["CT_MIP"] = df_of_collages["CT_MIP"].str.replace("SUV_MIP", "CT_MIP")
df_of_collages["CT_bone"] = df_of_collages["CT_bone"].str.replace("SUV_bone", "CT_bone")
df_of_collages["CT_lean"] = df_of_collages["CT_lean"].str.replace("SUV_lean", "CT_lean")
df_of_collages["CT_adipose"] = df_of_collages["CT_adipose"].str.replace("SUV_adipose", "CT_adipose")
df_of_collages["CT_air"] = df_of_collages["CT_air"].str.replace("SUV_air", "CT_air")


df_of_collages = df_of_collages[["patient_ID", "scan_date", "SUV_MIP", "CT_MIP", "SUV_bone", "CT_bone", "SUV_lean", "CT_lean", "SUV_adipose", "CT_adipose", "SUV_air", "CT_air"]]
df_of_collages = df_of_collages.drop_duplicates()
df_of_collages.to_excel("/media/andres/T7 Shield1/UCAN_project/df_of_raw_collages.xlsx", index=False)

