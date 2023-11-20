import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.utility.array import ToTensor
from monai.transforms.intensity.array import ScaleIntensity

class ImageDataset(Dataset):
    def __init__(self, SUV_MIP_files, SUV_bone_files, SUV_lean_files, SUV_adipose_files, SUV_air_files, CT_MIP_files, CT_bone_files, CT_lean_files, CT_adipose_files, CT_air_files, labels):
        self.SUV_MIP_files = SUV_MIP_files
        self.SUV_bone_files = SUV_bone_files
        self.SUV_lean_files = SUV_lean_files
        self.SUV_adipose_files = SUV_adipose_files
        self.SUV_air_files = SUV_air_files
        self.CT_MIP_files = CT_MIP_files
        self.CT_bone_files = CT_bone_files
        self.CT_lean_files = CT_lean_files
        self.CT_adipose_files = CT_adipose_files
        self.CT_air_files = CT_air_files
        self.labels = labels
        self.transform = Compose(
            [
                LoadImage(image_only=True, dtype=float),
                ToTensor(),
                ScaleIntensity(minv=0, maxv=1)
            ]
        )
    
    def __len__(self):
        return len(self.SUV_MIP_files)
    
    def __getitem__(self, index):
        SUV_MIP_path = self.SUV_MIP_files[index]
        SUV_bone_path = self.SUV_bone_files[index]
        SUV_lean_path = self.SUV_lean_files[index]
        SUV_adipose_path = self.SUV_adipose_files[index]
        SUV_air_path = self.SUV_air_files[index]
        CT_MIP_path = self.CT_MIP_files[index]
        CT_bone_path = self.CT_bone_files[index]
        CT_lean_path = self.CT_lean_files[index]
        CT_adipose_path = self.CT_adipose_files[index]
        CT_air_path = self.CT_air_files[index]
        label = self.labels[index]

        # Load and transform the images
        SUV_MIP = self.transform(SUV_MIP_path)
        SUV_bone = self.transform(SUV_bone_path)
        SUV_lean = self.transform(SUV_lean_path)
        SUV_adipose = self.transform(SUV_adipose_path)
        SUV_air = self.transform(SUV_air_path)
        CT_MIP = self.transform(CT_MIP_path)
        CT_bone = self.transform(CT_bone_path)
        CT_lean = self.transform(CT_lean_path)
        CT_adipose = self.transform(CT_adipose_path)
        CT_air = self.transform(CT_air_path)

        # Concatenate the images along the channel dimension
        SUV_MIP_new = torch.unsqueeze(SUV_MIP, 0)
        SUV_bone_new = torch.unsqueeze(SUV_bone, 0)
        SUV_lean_new = torch.unsqueeze(SUV_lean, 0)
        SUV_adipose_new = torch.unsqueeze(SUV_adipose, 0)
        SUV_air_new = torch.unsqueeze(SUV_air, 0)
        CT_MIP_new = torch.unsqueeze(CT_MIP, 0)
        CT_bone_new = torch.unsqueeze(CT_bone, 0)
        CT_lean_new = torch.unsqueeze(CT_lean, 0)
        CT_adipose_new = torch.unsqueeze(CT_adipose, 0)
        CT_air_new = torch.unsqueeze(CT_air, 0)

        #multi_channel_input = torch.cat((SUV_MIP_new, SUV_bone_new, SUV_lean_new, SUV_adipose_new, SUV_air_new, CT_MIP_new, CT_bone_new, CT_lean_new, CT_adipose_new, CT_air_new), dim=0)
        multi_channel_input = torch.cat((SUV_MIP_new,SUV_MIP_new), dim=0)
        return multi_channel_input, label
    
def prepare_data(args, df_train, batch_size, shuffle=None, label=None):
    if shuffle==True:
        df_train_shuffled = df_train.sample(frac=1).reset_index(drop=True)
    else:
        df_train_shuffled = df_train
    
    SUV_MIP_train = df_train_shuffled["SUV_MIP"].tolist()
    SUV_bone_train = df_train_shuffled["SUV_bone"].tolist()
    SUV_lean_train = df_train_shuffled["SUV_lean"].tolist()
    SUV_adipose_train = df_train_shuffled["SUV_adipose"].tolist()
    SUV_air_train = df_train_shuffled["SUV_air"].tolist()
    CT_MIP_train = df_train_shuffled["CT_MIP"].tolist()
    CT_bone_train = df_train_shuffled["CT_bone"].tolist()
    CT_lean_train = df_train_shuffled["CT_lean"].tolist()
    CT_adipose_train = df_train_shuffled["CT_adipose"].tolist()
    CT_air_train = df_train_shuffled["CT_air"].tolist()
    
    if label == "sex":
        label_train = df_train_shuffled["sex"].tolist()
    elif label == "diagnosis":
        label_train = df_train_shuffled["diagnosis"].tolist()
    elif label == "patient_age":
        label_train = df_train_shuffled["patient_age"].tolist()
    elif label == "MTV":
        label_train = df_train_shuffled["MTV (ml)"].tolist()
    elif label == "lean_volume":
        label_train = df_train_shuffled["lean_volume (L)"].tolist()
    elif label == "lesion_count":
        label_train = df_train_shuffled["lesion_count"].tolist()
    else:
        label_train = df_train_shuffled["age"].tolist()
     
    train_files = [
        {"SUV_MIP": SUV_MIP_name, "SUV_bone": SUV_bone_name, "SUV_lean": SUV_lean_name, "SUV_adipose": SUV_adipose_name, "SUV_MIP": SUV_air_name, 
         "CT_MIP": CT_MIP_name, "CT_bone": CT_bone_name, "CT_lean": CT_lean_name, "CT_adipose": CT_adipose_name, "CT_air": CT_air_name, "label": label_name}
         for SUV_MIP_name, SUV_bone_name, SUV_lean_name, SUV_adipose_name, SUV_air_name, CT_MIP_name, CT_bone_name, CT_lean_name, CT_adipose_name, CT_air_name, label_name in
         zip(SUV_MIP_train, SUV_bone_train, SUV_lean_train, SUV_adipose_train, SUV_air_train, CT_MIP_train, CT_bone_train, CT_lean_train, CT_adipose_train, CT_air_train, label_train)
    ]

    train_ds = ImageDataset(SUV_MIP_train, SUV_bone_train, SUV_lean_train, SUV_adipose_train, SUV_air_train, CT_MIP_train, CT_bone_train, CT_lean_train, CT_adipose_train, CT_air_train, label_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=args["num_workers"])

    return train_files, train_loader