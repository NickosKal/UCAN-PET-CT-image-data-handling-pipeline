import os
import shutil
from tabnanny import verbose
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from monai.apps.utils import download_and_extract
from monai.config.deviceconfig import print_config
from monai.data.utils import decollate_batch 
from monai.data.dataloader import DataLoader
from monai.data.image_dataset import ImageDataset
from monai.metrics.rocauc import ROCAUCMetric
from monai.networks.nets.densenet import DenseNet121
from tqdm import tqdm

from monai.transforms.post.array import Activations, AsDiscrete
from monai.transforms.utility.array import EnsureChannelFirst
from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.spatial.array import RandFlip,RandRotate,RandZoom
from monai.transforms.intensity.array import ScaleIntensity

from monai.utils.misc import set_determinism
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
from tqdm import tqdm
from generate_dataset import prepare_data

import sys

parent_dir = os.path.abspath('../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Task4.utils import plot, train_regression, validation_regression

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import densenet121

experiment = "4"
k_fold = 10
learning_rate = 1e-4
weight_decay = 1e-5
batch_size_train = 2
args = {"num_workers": 4,
        "batch_size_val": 1} #25

#df = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/dataset_for_training_regression.xlsx")
df = pd.read_excel("/home/ashish/Ashish/UCAN/dataset_for_training_regression_v2.xlsx")
df_sorted = df.sort_values(by="patient_ID")

try:
    df_clean = df_sorted.drop(columns="Unnamed: 0").reset_index(drop=True)
except:
    df_clean = df_sorted.copy()

#path_output = "/media/andres/T7 Shield1/UCAN_project/Results/regression"
path_output = "/home/ashish/Ashish/UCAN/Results/regression/experiment_" + experiment + "/"
outcome = "patient_age" # "mtv"

checkpoint_path = "/home/ashish/Ashish/UCAN/Results/regression/experiment_4/CV_0/Network_Weights/best_model_20.pth.tar"

pre_trained_weights = True

class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(WideBasic, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class WideResNetRegression(nn.Module):
    def __init__(self, widen_factor=1, depth=16, num_channels=10, dropout_rate=0.25):
        super(WideResNetRegression, self).__init__()
        self.in_planes = 16
        k = widen_factor

        # Network architecture
        n = (depth - 4) // 6
        block = WideBasic

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(block, 16*k, n, stride=1)
        self.layer2 = self._wide_layer(block, 32*k, n, stride=2)
        self.layer3 = self._wide_layer(block, 64*k, n, stride=2)
        self.bn1 = nn.BatchNorm2d(64*k, momentum=0.9)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer with the specified rate
        self.linear = nn.Linear(64*k, 1)  # Output a single value for regression

    def _wide_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn1(out)
        out = torch.mean(out, dim=(2, 3))  # Global average pooling
        out = self.dropout(out)
        out = self.linear(out)
        return out

for k in tqdm(range(k_fold)):
    if k >= 0:
        print(f"Cross validation for fold {k}")
        max_epochs = 500
        val_interval = 1  #5
        best_metric = 100000000000 #1000000
        best_metric_epoch = -1
        metric_values = []
        metric_values_r_squared = []
        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(torch.cuda.is_available())
        print(device)

        #model = DenseNet121(spatial_dims=2, in_channels=10, out_channels=1, dropout_prob=0.25).to(device)
        model = WideResNetRegression(widen_factor=8, depth=16, num_channels=10, dropout_rate=0.25).to(device)
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True).to(device)
        
        if pre_trained_weights:
            # Use it in case we have pre trained weights
            print("Checkpoint Loading for Cross Validation: {}".format(k))
            #checkpoint_path = load_checkpoint(args, k)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['net'])
            pass
        else:
            print("Training from Scratch!") 

        loss_function = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        if not os.path.exists(path_output+"CV_"+str(k)+'/Network_Weights/'):
            os.makedirs(path_output+"CV_"+str(k)+'/Network_Weights/')

        if not os.path.exists(path_output+"CV_"+str(k)+'/Metrics/'):
            os.makedirs(path_output+"CV_"+str(k)+'/Metrics/')
        
        if not os.path.exists(path_output+"CV_"+str(k)+'/MIPs/'):
            os.makedirs(path_output+"CV_"+str(k)+'/MIPs/')

        #os.mkdir("dir path", k)

        # factor = round(df.shape[0]/k_fold)
        # if k == (k_fold - 1):
        #     df_val = df[factor*k:].reset_index(drop=True)
        # else:
        #     df_val = df[factor*k:factor*k+factor].reset_index(drop=True)
        # df_train = df[~df.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

        #patients_for_train = df_clean[:int(df_clean.shape[0] * 0.7)].patient_ID.tolist()
        #df_train = df_clean[df_clean.patient_ID.isin(patients_for_train)]
        #df_val = df_clean[~df_clean.patient_ID.isin(patients_for_train)]

        factor = round(df.shape[0]/k_fold)
        if k == (k_fold - 1):
            patients_for_val = df_clean[factor*k:].patient_ID.tolist()
            df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)
        else:
            patients_for_val = df_clean[factor*k:factor*k+factor].patient_ID.tolist()
            df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

        df_train = df_clean[~df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

        print("Number of patients in Training set: ", len(df_train))
        print("Number of patients in Valdation set: ", len(df_val))

        train_files, train_loader = prepare_data(args, df_train, batch_size_train, shuffle=False, label=outcome)

        train_loss = []
        for epoch in tqdm(range(max_epochs)):
            epoch += 21
            #Training
            epoch_loss, train_loss = train_regression(model, train_files, train_loader, optimizer, loss_function, device, train_loss)
            print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")

            #Validation
            if (epoch + 1) % val_interval == 0:
                metric_values, best_metric_new = validation_regression(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, metric_values_r_squared, path_output, outcome, loss_function)

                best_metric = best_metric_new

            #Save and plot
            np.save(os.path.join(path_output, "CV_" + str(k) + "/MAE.npy"), metric_values)

            path_MAE = os.path.join(path_output, "CV_" + str(k), "epoch_vs_MAE.jpg")

            if len(metric_values) > 2:
                plot(metric_values, path_MAE, "MAE")
                #plot(metric_values_r_squared, path_r_squared, "R2")
