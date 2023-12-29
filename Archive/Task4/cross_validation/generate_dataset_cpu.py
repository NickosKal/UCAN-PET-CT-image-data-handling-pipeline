import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.utility.array import ToTensor
from monai.transforms.intensity.array import ScaleIntensity

class ImageDataset(Dataset):
    def __init__(self, SUV_MIP_files,CT_MIP_files,labels):
        self.SUV_MIP_files = SUV_MIP_files

        self.CT_MIP_files = CT_MIP_files

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

        CT_MIP_path = self.CT_MIP_files[index]

        label = self.labels[index]

        # Load and transform the images
        SUV_MIP = self.transform(SUV_MIP_path)

        CT_MIP = self.transform(CT_MIP_path)


        # Concatenate the images along the channel dimension
        SUV_MIP_new = torch.unsqueeze(SUV_MIP, 0)

        CT_MIP_new = torch.unsqueeze(CT_MIP, 0)


        multi_channel_input = torch.cat((SUV_MIP_new, CT_MIP_new), dim=0)
        #multi_channel_input = torch.cat((SUV_MIP_new,SUV_MIP_new), dim=0)
        return multi_channel_input, label
    
def prepare_data(args, df_train, batch_size, shuffle=None, label=None):
    if shuffle==True:
        df_train_shuffled = df_train.sample(frac=1).reset_index(drop=True)
    else:
        df_train_shuffled = df_train
    
    SUV_MIP_train = df_train_shuffled["SUV_MIP"].tolist()

    CT_MIP_train = df_train_shuffled["CT_MIP"].tolist()

    
    if label == "sex":
        label_train = df_train_shuffled["sex"].tolist()
    elif label == "GT_diagnosis_label":
        label_train = df_train_shuffled["GT_diagnosis_label"].tolist()
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
        {"SUV_MIP": SUV_MIP_name,  
         "CT_MIP": CT_MIP_name, "label": label_name}
         for SUV_MIP_name, CT_MIP_name, label_name in
         zip(SUV_MIP_train, CT_MIP_train, label_train)
    ]

    train_ds = ImageDataset(SUV_MIP_train, CT_MIP_train, label_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=args["num_workers"])

    return train_files, train_loader