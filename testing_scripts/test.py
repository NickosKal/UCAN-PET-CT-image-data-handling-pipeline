import numpy as np
import os
import pandas as pd
from tqdm import tqdm

reshaped_projections_path = "/media/andres/T7 Shield1/UCAN_project/2D_projections/reshaped/"
df_of_raw_projections = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/df_of_raw_projections.xlsx")
df_of_raw_projections_test = df_of_raw_projections.head(4)

cropped_array = np.zeros((580, 256))

# Initialize a loop to start the reshaping process
for row, image in df_of_raw_projections_test.iterrows():

    save_path_temp = os.path.join(reshaped_projections_path, str(image["patient_ID"]), str(image["scan_date"]))
    if not os.path.exists(save_path_temp):
        os.makedirs(save_path_temp)

    SUV_MIP = np.load(image["SUV_MIP"])
    size = SUV_MIP.shape[0]

    # Check if the size of the array is lower than 580 if yes apply 0 padding.
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
        result = np.pad(SUV_bone, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_bone" + ".npy"), result)

        SUV_lean = np.load(image["SUV_lean"])
        result = np.pad(SUV_lean, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_lean" + ".npy"), result)

        SUV_adipose = np.load(image["SUV_adipose"])
        result = np.pad(SUV_adipose, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_adipose" + ".npy"), result)

        SUV_air = np.load(image["SUV_air"])
        result = np.pad(SUV_air, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "SUV_air" + ".npy"), result)
        
        CT_MIP = np.load(image["CT_MIP"])
        result = np.pad(SUV_MIP, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_MIP" + ".npy"), result)

        CT_bone = np.load(image["CT_bone"])
        result = np.pad(CT_bone, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_bone" + ".npy"), result)

        CT_lean = np.load(image["CT_lean"])
        result = np.pad(CT_lean, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_lean" + ".npy"), result)

        CT_adipose = np.load(image["CT_adipose"])
        result = np.pad(CT_adipose, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_adipose" + ".npy"), result)

        CT_air = np.load(image["CT_air"])
        result = np.pad(CT_air, ((pad_from_top, pad_from_bottom), (0,0)))
        np.save(os.path.join(save_path_temp, "CT_air" + ".npy"), result)

    # If the size is bigger than 580 crop the array by coppying the bigger array to the array of 0 that has been created in the beginning of the script.
    else:
        temp_crop_int = (size - 580)//2
        temp_crop_float = (size - 580)/2
        crop_from_top = temp_crop_int
        crop_from_bottom = crop_from_top

        if temp_crop_int < temp_crop_float:
            crop_from_top = temp_crop_int + 1

        cropped_array = SUV_MIP[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_MIP" + ".npy"), cropped_array)

        SUV_bone = np.load(image["SUV_bone"])
        cropped_array = SUV_bone[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_bone" + ".npy"), cropped_array)

        SUV_lean = np.load(image["SUV_lean"])
        cropped_array = SUV_lean[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_lean" + ".npy"), cropped_array)

        SUV_adipose = np.load(image["SUV_adipose"])
        cropped_array = SUV_adipose[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_adipose" + ".npy"), cropped_array)

        SUV_air = np.load(image["SUV_air"])
        cropped_array = SUV_air[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "SUV_air" + ".npy"), cropped_array)
        
        CT_MIP = np.load(image["CT_MIP"])
        cropped_array = CT_MIP[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_MIP" + ".npy"), cropped_array)

        CT_bone = np.load(image["CT_bone"])
        cropped_array = CT_bone[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_bone" + ".npy"), cropped_array)

        CT_lean = np.load(image["CT_lean"])
        cropped_array = CT_lean[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_lean" + ".npy"), cropped_array)

        CT_adipose = np.load(image["CT_adipose"])
        cropped_array = CT_adipose[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_adipose" + ".npy"), cropped_array)

        CT_air = np.load(image["CT_air"])
        cropped_array = CT_air[crop_from_top:-crop_from_bottom, :]
        np.save(os.path.join(save_path_temp, "CT_air" + ".npy"), cropped_array)

# Create a dataframe with the paths of the reshaped projections.
# df_of_reshaped_projections = pd.DataFrame(columns=["patient_ID", "scan_date", "SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air"])
# for patient_ID in tqdm(sorted(os.listdir(reshaped_projections_path))):
#     for scan_date in sorted(os.listdir(os.path.join(reshaped_projections_path, patient_ID))):
#         for angle in ["-90.0", "0.0"]:
#             SUV_MIP_path = os.path.join(reshaped_projections_path, patient_ID, scan_date, "SUV_MIP" + ".npy")
#             SUV_bone_path = os.path.join(reshaped_projections_path, patient_ID, scan_date, "SUV_bone" + ".npy")
#             SUV_lean_path = os.path.join(reshaped_projections_path, patient_ID, scan_date, "SUV_lean" + ".npy")
#             SUV_adipose_path = os.path.join(reshaped_projections_path, patient_ID, scan_date, "SUV_adipose" + ".npy")
#             SUV_air_path = os.path.join(reshaped_projections_path, patient_ID, scan_date, "SUV_air" + ".npy")
#             df_temp = pd.DataFrame({"patient_ID": [patient_ID], "scan_date": [scan_date], "SUV_MIP": [SUV_MIP_path], "SUV_bone": [SUV_bone_path], "SUV_lean": [SUV_lean_path], "SUV_adipose": [SUV_adipose_path], "SUV_air": [SUV_air_path]})
#             df_of_reshaped_projections = pd.concat([df_of_reshaped_projections, df_temp], ignore_index=True)

# df_of_reshaped_projections["CT_MIP"] = df_of_reshaped_projections["SUV_MIP"]
# df_of_reshaped_projections["CT_bone"] = df_of_reshaped_projections["SUV_bone"]
# df_of_reshaped_projections["CT_lean"] = df_of_reshaped_projections["SUV_lean"]
# df_of_reshaped_projections["CT_adipose"] = df_of_reshaped_projections["SUV_adipose"]
# df_of_reshaped_projections["CT_air"] = df_of_reshaped_projections["SUV_air"]

# df_of_reshaped_projections["CT_MIP"] = df_of_reshaped_projections["CT_MIP"].str.replace("SUV_MIP", "CT_MIP")
# df_of_reshaped_projections["CT_bone"] = df_of_reshaped_projections["CT_bone"].str.replace("SUV_bone", "CT_bone")
# df_of_reshaped_projections["CT_lean"] = df_of_reshaped_projections["CT_lean"].str.replace("SUV_lean", "CT_lean")
# df_of_reshaped_projections["CT_adipose"] = df_of_reshaped_projections["CT_adipose"].str.replace("SUV_adipose", "CT_adipose")
# df_of_reshaped_projections["CT_air"] = df_of_reshaped_projections["CT_air"].str.replace("SUV_air", "CT_air")


# df_of_reshaped_projections = df_of_reshaped_projections[["patient_ID", "scan_date", "SUV_MIP", "CT_MIP", "SUV_bone", "CT_bone", "SUV_lean", "CT_lean", "SUV_adipose", "CT_adipose", "SUV_air", "CT_air"]]
# df_of_reshaped_projections.to_excel("/media/andres/T7 Shield1/UCAN_project/df_of_reshaped_projections.xlsx", index=False)