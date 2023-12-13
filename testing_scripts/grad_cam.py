import numpy as np
from monai.networks.nets import DenseNet121
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImage, ToTensor, ScaleIntensity
from tqdm import tqdm
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNet121(spatial_dims=2, in_channels=10,
                    out_channels=1, dropout_prob=0.0).cuda()

#checkpoint_path = "/media/andres/T7 Shield1/UCAN_project/Results/regression/Experiment_1/CV_0/Network_Weights/best_model_847.pth.tar"
checkpoint_path = "/home/ashish/Ashish/UCAN/ReshapedCollages/bestmodel_ageprediction/best_model_847.pth.tar"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['net'])

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

        multi_channel_input = torch.cat((SUV_MIP_new, SUV_bone_new, SUV_lean_new, SUV_adipose_new, SUV_air_new, CT_MIP_new, CT_bone_new, CT_lean_new, CT_adipose_new, CT_air_new), dim=0)
        #multi_channel_input = torch.cat((CT_MIP_new, CT_bone_new, CT_lean_new, CT_adipose_new, CT_air_new), dim=0)
        #multi_channel_input = torch.cat((SUV_MIP_new, SUV_bone_new, SUV_lean_new, SUV_adipose_new, SUV_air_new), dim=0)
        #multi_channel_input = torch.cat((SUV_MIP_new, SUV_MIP_new))

        return multi_channel_input, label


#Regression
# Define Grad-CAM class
class GradCAM:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.gradients = []

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        for layer in self.layers:
            layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output.backward()

        heatmaps = []
        for gradient in self.gradients:
            pooled_gradient = torch.mean(gradient, dim=[2, 3], keepdim=True)
            heatmap = (pooled_gradient * gradient).sum(dim=1, keepdim=True)
            heatmap = nn.functional.relu(heatmap)
            heatmap /= heatmap.max()
            heatmaps.append(heatmap)

        return heatmaps


def prepare_data(df_train, batch_size, shuffle=None, label=None):
    if shuffle==True:
        df_train_shuffled = df_train.sample(frac=1).reset_index(drop=True)
    elif shuffle==False:
        df_train_shuffled = df_train

    SUV_MIP_train = df_train_shuffled['SUV_MIP'].tolist()
    SUV_bone_train = df_train_shuffled['SUV_bone'].tolist()
    SUV_lean_train = df_train_shuffled['SUV_lean'].tolist()
    SUV_adipose_train = df_train_shuffled['SUV_adipose'].tolist()
    SUV_air_train = df_train_shuffled['SUV_air'].tolist()
    CT_MIP_train = df_train_shuffled['CT_MIP'].tolist()
    CT_bone_train = df_train_shuffled['CT_bone'].tolist()
    CT_lean_train = df_train_shuffled['CT_lean'].tolist()
    CT_adipose_train = df_train_shuffled['CT_adipose'].tolist()
    CT_air_train = df_train_shuffled['CT_air'].tolist()

    if label == "sex":
        label_train = df_train_shuffled['sex'].tolist()
    if label == "diagnosis":
        label_train = df_train_shuffled['diagnosis'].tolist()
    elif label == "patient_age":
        label_train = df_train_shuffled['patient_age'].tolist()
    elif label == "MTV":
        label_train = df_train_shuffled['MTV (ml)'].tolist()
    elif label == "lean_volume":
        label_train = df_train_shuffled['lean_volume (L)'].tolist()

    train_files = [
        {"SUV_MIP": SUV_MIP_name, "SUV_bone": SUV_bone_name, "SUV_lean": SUV_lean_name, "SUV_adipose": SUV_adipose_name, "SUV_air": SUV_air_name, 
        "CT_MIP": CT_MIP_name, "CT_bone": CT_bone_name, "CT_lean": CT_lean_name, "CT_adipose": CT_adipose_name, "CT_air": CT_air_name, "label": label_name}
        for SUV_MIP_name, SUV_bone_name, SUV_lean_name, SUV_adipose_name, SUV_air_name, CT_MIP_name, CT_bone_name, CT_lean_name, CT_adipose_name, CT_air_name, label_name in 
        zip(SUV_MIP_train, SUV_bone_train, SUV_lean_train, SUV_adipose_train, SUV_air_train, CT_MIP_train, CT_bone_train, CT_lean_train, CT_adipose_train, CT_air_train, label_train)
    ]

    train_ds = ImageDataset(SUV_MIP_train, SUV_bone_train, SUV_lean_train, SUV_adipose_train, SUV_air_train, CT_MIP_train, CT_bone_train, CT_lean_train, CT_adipose_train, CT_air_train, label_train)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    return train_files, train_loader


def generate_heatmap(patient_ID):

    layers_to_visualize = [model.features.transition3, model.features.transition2,
                       model.features.transition1, model.features.conv0]
    
    grad_cam = GradCAM(model, layers_to_visualize)

    val_files, val_loader = prepare_data(patient_ID)
    heatmaps = grad_cam.generate_heatmap(inputs)


K = 10
k = 9

df = pd.read_excel("/home/ashish/Ashish/UCAN/ReshapedCollages/Files/dataset_for_model_regression_training.xlsx")
df = df.replace('/media/andres/T7 Shield1/UCAN_project/collages/reshaped_collages', '/home/ashish/Ashish/UCAN/ReshapedCollages/collages', regex=True)

factor = round(df.shape[0]/K)
if k == (K - 1):
    df_val = df[factor*k:].reset_index(drop=True)
else:
    df_val = df[factor*k:factor*k+factor].reset_index(drop=True)
df_train = df[~df.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

df_train_new = df[df.scan_date.isin(df_train.scan_date)].reset_index(drop=True)
df_val_new = df[df.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

scan_dates = np.unique(df_val_new["scan_date"])
np.random.shuffle(scan_dates)

for scan_date in tqdm(scan_dates):
    df_temp = df_val_new[df_val_new["scan_date"]==scan_date].reset_index(drop=True)
    #df_temp = df_val_new[df_val_new["scan_date"]=="03-27-2005-NA-PET-CT Ganzkoerper  primaer mit KM-38725"].reset_index(drop=True)
    
    pat_id = np.unique(df_temp["patient_ID"])
    #val_files, val_loader = prepare_data(args, df_temp, shuffle=False, label="age")
    val_files, val_loader = prepare_data(df_temp, 1, shuffle=True, label="patient_age")
    
    for inputs, labels in val_loader:
        model.eval()
        inputs, labels = inputs.cuda(), labels.cuda()


layers_to_visualize = [model.features.transition3, model.features.transition2,
                       model.features.transition1, model.features.conv0]

# Create Grad-CAM instance
grad_cam = GradCAM(model, layers_to_visualize)

# Generate and visualize heatmaps for each layer
heatmaps = grad_cam.generate_heatmap(inputs)

# Resize heatmaps to match the original image size
resized_heatmaps = []
for heatmap in heatmaps:
    resized_heatmap = nn.functional.interpolate(heatmap, size=(512, 512), mode='bilinear', align_corners=False)
    resized_heatmaps.append(resized_heatmap)

i = inputs[0,0,:,:].data.cpu().numpy()
h = resized_heatmaps[1][0,0,:,:].data.cpu().numpy()

# Normalize the heatmap values
heatmap_normalized = h / np.max(h)

# Set a transparency factor for the heatmap overlay
alpha = 0.7

# Overlay the heatmap on the image using element-wise addition and transparency
overlayed_image = alpha * heatmap_normalized + (1 - alpha) * i

# Clip values to stay within [0, 1] range
overlayed_image = np.clip(overlayed_image, 0, 1)

output_path = "/media/sambit/HDD/Sambit/Projects/Project_6/Data_Analysis/gradcam-mathod/Output/Age"
save_path = os.path.join(output_path, pat_id[0])

if not os.path.exists(save_path):
    os.makedirs(save_path)

plt.imshow(overlayed_image, cmap="coolwarm")
#plt.imshow(overlayed_image, cmap="RdYlBu_r")

plt.savefig(save_path + "/overlap.jpg", dpi=400)
#plt.show()
#plt.imshow(i, cmap="gray")

plt.imshow(i, cmap="gray")
#plt.imshow(overlayed_image, cmap="RdYlBu_r")

plt.savefig(save_path + "/img.jpg", dpi=400)
print(pat_id)

