import pandas as pd
import numpy as np

def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)


classification_data = pd.read_excel("/home/ashish/Ashish/UCAN/ReshapedCollages/dataset_for_model_training.xlsx")
classification_data.shape

classification_data.patient_ID = classification_data.patient_ID.str.replace(" ","")
display_full(classification_data.head(2))

#clinicaldata_path = "/home/ashish/Ashish/UCAN/U-CAN Lymfom - Ahlström 211021 - utlämnat_pnr_removed.xlsx"
clinicaldata_path = "/home/ashish/Ashish/UCAN/20231129_UCAN_lym_export/Lymphoma_Octopus_Export 2023-11-29_C80_only_n1291_incl_YYYY_MM_pnr_removed.xlsx"
clinical_data = pd.read_excel(clinicaldata_path, header=0)
print(clinical_data.columns)
clinical_data = clinical_data.sort_values("personReference")
clinical_data = clinical_data[["personReference", "BirthYear", "BirthMonth", "consentDate", "translatedDiagnosis", "sex"]].copy()
clinical_data.columns = clinical_data.columns.str.strip()
clinical_data.npr = clinical_data.personReference.str.replace(" ","")
print(clinical_data.shape)
print(clinical_data.npr.nunique())
display_full(clinical_data.head(2))

classification_data_v1 = pd.merge(classification_data, clinical_data, how="inner", left_on=["patient_ID"], right_on=["personReference"], suffixes=['','_'])
print(classification_data_v1.shape)
display_full(classification_data_v1.head(2))

classification_data_v1["diagnosis_groups"] = classification_data_v1["translatedDiagnosis"].apply(lambda x: x[:3])
#["diaGroups_GT"] = np.where(classification_data_v1["diaGroups_new"]=="C83", 1, np.where(classification_data_v1["diaGroups_new"]=="C81", 2, 3))

classification_data_v1["patientDOB"] = classification_data_v1.apply(lambda x: str(x["BirthYear"]) + (str(x["BirthMonth"]) if len(str(x["BirthMonth"]))>1 else "0" + str(x["BirthMonth"])) + "15", axis=1)
classification_data_v1[["scan_date", "patientDOB"]][:5]

from datetime import date

classification_data_v1["scan_date"] = pd.to_datetime(classification_data_v1["scan_date"], format="%Y%m%d")
classification_data_v1["patientDOB"] = pd.to_datetime(classification_data_v1["patientDOB"], format="%Y%m%d")
classification_data_v1["consentDate"] = pd.to_datetime(classification_data_v1["consentDate"], format="%Y-%m-%d")
classification_data_v1[["patientDOB", "scan_date", "consentDate"]][:5]

def calculateAge(dob, scan_date):
    return scan_date.year - dob.year - ((scan_date.month, scan_date.day) < (dob.month, dob.day))

classification_data_v1["age"] = classification_data_v1.apply(lambda x: calculateAge(x["patientDOB"], x["scan_date"]), axis=1)
classification_data_v1[["patientDOB", "scan_date", "age"]][:5]

classification_data_v1["diff_date (abs(sd-cd))"] = classification_data_v1.apply(lambda x: abs((x["scan_date"] - x["consentDate"])/np.timedelta64(1, 'D')), axis=1)

classification_data_v2 = classification_data_v1[['patient_ID', 'patientDOB', 'scan_date', 'consentDate', 'diff_date (abs(sd-cd))', 'sex', 'patient_age', 'diagnosis_groups', 
                                                 'translatedDiagnosis', 'SUV_MIP', 'CT_MIP', 'SUV_bone', 'CT_bone', 'SUV_lean', 'CT_lean', 'SUV_adipose', 
                                                 'CT_adipose', 'SUV_air', 'CT_air']].copy()
classification_data_v2 = classification_data_v2.rename(columns={'patientDOB':'dob', 'patient_age':'age', 'consentDate':'consent_date', 'translatedDiagnosis':'diagnosis'})
classification_data_v2 = classification_data_v2.sort_values(["patient_ID", "scan_date"])
display_full(classification_data_v2.head(4))

classification_data_v2.to_excel("/home/ashish/Ashish/UCAN/dataset_for_training_366patients_clinical20231129.xlsx", index=False)

### select baseline scans with closest date to consent date
classification_data_v3 = classification_data_v2.copy()
display_full(classification_data_v3.head(3))

classification_data_v3["rank"] = classification_data_v3.groupby("patient_ID")["diff_date (abs(sd-cd))"].rank(method="min", ascending=True)
display_full(classification_data_v3[['patient_ID', 'dob', 'scan_date', 'consent_date', 'diff_date (abs(sd-cd))', "rank"]].sort_values(["patient_ID","rank"]))

df1 = classification_data_v3[classification_data_v3["rank"]==1].copy()
df1.drop(columns="rank", inplace=True)
df1.shape

nan_df = classification_data_v3[classification_data_v3["rank"].isin([np.nan])].copy()
nan_df["rank1"] = nan_df.groupby("patient_ID")["scan_date"].rank(method="first", ascending=False)
nan_df[["patient_ID", "scan_date", "rank1"]]

df2 = nan_df[nan_df["rank1"]==1].copy()
df2.drop(columns=["rank", "rank1"], inplace=True)
df2.head(2)

classification_data_v4 = pd.concat([df1,df2], ignore_index=True).reset_index(drop=True)
print(classification_data_v4.shape)
display_full(classification_data_v4.head())

classification_data_v4.to_excel("/home/ashish/Ashish/UCAN/dataset_for_training_366patients_baseline_scans_clinical20231129.xlsx", index=False)