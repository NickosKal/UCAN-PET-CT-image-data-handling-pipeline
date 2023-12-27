import os
import shutil
from tabnanny import verbose
import tempfile
from typing import Sequence
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
from torchvision.models import efficientnet_b7, EfficientNet

from monai.utils.misc import set_determinism
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
from tqdm import tqdm
#from generate_dataset import prepare_data
from generate_dataset_cpu import prepare_data

import sys

parent_dir = os.path.abspath('../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Task4.utils import plot, train_regression, validation_regression, working_system

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import densenet121

from Utils import utils

# reading main config file
config = utils.read_config()

system = 2 # 1 or 2
if system == 1:
    PATH = config["Source"]["paths"]["source_path_system_1"]
elif system == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    PATH = config["Source"]["paths"]["source_path_system_2"]
    working_system(system)
else:
    PATH = ""
    print("Invalid system")

experiment = "3"
k_fold = 10
learning_rate = 1e-4
weight_decay = 1e-5
batch_size_train = 14
args = {"num_workers": 4,
        "batch_size_val": 1} #25

#efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b7', pretrained=True)

#df = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/dataset_for_training_regression.xlsx")
#df = pd.read_excel("/home/ashish/Ashish/UCAN/dataset_for_training_regression_v2.xlsx")
df_path = PATH + config['collages_for_rergession_dataframe']
df = pd.read_excel(df_path)
#df = pd.read_excel("//home/ashish/Ashish/UCAN/ReshapedCollages/dataset_for_model_training_v1.xlsx")

df = df.sort_values(by="scan_date")
df['scan_date'] = df['scan_date'].astype(str)
df['unique_patient_ID_scan_date'] = df['patient_ID'] + '_' + df['scan_date']
df = df.drop(columns=['patient_ID', 'scan_date'])

df = df.sort_values(by="unique_patient_ID_scan_date")

path_output = PATH + config['regression_path']
output_path = os.path.join(path_output, "Experiment_" + str(experiment) + "/")

#path_output = "/media/andres/T7 Shield1/UCAN_project/Results/regression"
#path_output = "/home/ashish/Ashish/UCAN/Results/regression/experiment_" + experiment + "/"

outcome = "patient_age" # "mtv"

#heckpoint_path = "/home/ashish/Ashish/UCAN/Results/regression/experiment_6/CV_0/Network_Weights/best_model_46.pth.tar"

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
        max_epochs = 250
        val_interval = 1  #5
        best_metric = 1000000#7.379 #1000000
        best_metric_epoch = -1#46#-1
        metric_values = []
        metric_values_r_squared = []
        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(torch.cuda.is_available())
        print(device)
        #model = EfficientNet(inverted_residual_setting = 'Sequence', input_channels=2, out_channels=1, dropout=0.25).to(device)
        model = DenseNet121(spatial_dims=2, in_channels=2, out_channels=1, dropout_prob=0.25).to(device)
        #model = WideResNetRegression(widen_factor=8, depth=22, num_channels=10, dropout_rate=0.25).to(device)
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True).to(device)
        
        if pre_trained_weights:
            # Use it in case we have pre trained weights
            print("Checkpoint Loading for Cross Validation: {}".format(k))
            checkpoint_path, epoch_to_continue = utils.load_checkpoints(system, "regression", None, experiment, k)
            #checkpoint_path = load_checkpoint(args, k)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['net'])
            #pass
        else:
            print("Training from Scratch!") 

        loss_function = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        if not os.path.exists(output_path+"CV_"+str(k)+'/Network_Weights/'):
            os.makedirs(output_path+"CV_"+str(k)+'/Network_Weights/')

        if not os.path.exists(output_path+"CV_"+str(k)+'/Metrics/'):
            os.makedirs(output_path+"CV_"+str(k)+'/Metrics/')
        
        if not os.path.exists(output_path+"CV_"+str(k)+'/MIPs/'):
            os.makedirs(output_path+"CV_"+str(k)+'/MIPs/')

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
        """"
        factor = round(df.shape[0]/k_fold)
        if k == (k_fold - 1):
            patients_for_val = df_clean[factor*k:].patient_ID.tolist()
            df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)
        else:
            patients_for_val = df_clean[factor*k:factor*k+factor].patient_ID.tolist()
            df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

        df_train = df_clean[~df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)
        """
        factor = round(df.shape[0]/k_fold)
        if k == (k_fold - 1):
            patients_for_val = df[factor*k:].unique_patient_ID_scan_date.tolist()
            df_val = df[df.unique_patient_ID_scan_date.isin(patients_for_val)].reset_index(drop=True)
        else:
            patients_for_val = df[factor*k:factor*k+factor].unique_patient_ID_scan_date.tolist()
            df_val = df[df.unique_patient_ID_scan_date.isin(patients_for_val)].reset_index(drop=True)

        df_train = df[~df.unique_patient_ID_scan_date.isin(patients_for_val)].reset_index(drop=True)

        print("Number of patients in Training set: ", len(df_train))
        print("Number of patients in Valdation set: ", len(df_val))

        train_files, train_loader = prepare_data(args, df_train, batch_size_train, shuffle=False, label=outcome)

        train_loss = []
        for epoch in tqdm(range(max_epochs)):
            #epoch += 47
            #Training
            if pre_trained_weights:
                    epoch = epoch + int(epoch_to_continue) + 1
            else:
                pass

            epoch_loss, train_loss = train_regression(model, train_files, train_loader, optimizer, loss_function, device, train_loss)
            print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")

            #Validation
            if (epoch + 1) % val_interval == 0:
                metric_values, best_metric_new = validation_regression(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, metric_values_r_squared, output_path, outcome, loss_function)

                best_metric = best_metric_new

            #Save and plot
            np.save(os.path.join(output_path, "CV_" + str(k) + "/MAE.npy"), metric_values)

            path_MAE = os.path.join(output_path, "CV_" + str(k), "epoch_vs_MAE.jpg")

            if len(metric_values) > 2:
                plot(metric_values, path_MAE, "MAE")
                #plot(metric_values_r_squared, path_r_squared, "R2")
