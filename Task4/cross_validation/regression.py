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

k_fold = 10
learning_rate = 1e-3
weight_decay = 0.001
batch_size_train = 10
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
path_output = "/home/ashish/Ashish/UCAN/Results/regression/"
outcome = "patient_age" # "mtv"
pre_trained_weights = False

for k in tqdm(range(k_fold)):
    if k >= 0:
        print(f"Cross validation for fold {k}")
        max_epochs = 20
        val_interval = 1 #5
        best_metric = 100000000000 #1000000
        best_metric_epoch = -1
        metric_values = []
        metric_values_r_squared = []
        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(torch.cuda.is_available())
        print(device)
        model = DenseNet121(spatial_dims=2, in_channels=10, out_channels=1, dropout_prob=0.25).to(device)
        
        if pre_trained_weights:
            # Use it in case we have pre trained weights
            # print("Checkpoint Loading for Cross Validation: {}".format(k))
            # checkpoint_path = load_checkpoint(args, k)
            # checkpoint = torch.load(checkpoint_path)
            # model.load_state_dict(checkpoint['net'])
            pass
        else:
            print("Training from Scratch!") 

        loss_function = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        if not os.path.exists(path_output+"CV_"+str(k)+'/Network_Weights/'):
            os.makedirs(path_output+"CV_"+str(k)+'/Network_Weights/')
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
