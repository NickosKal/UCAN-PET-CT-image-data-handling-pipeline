import os
import shutil
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
from generate_dataset import prepare_data

import sys
from utils import plot, train_regression, validation_regression

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import densenet121

k_fold = 15
learning_rate = 0
weight_decay = 0.001
batch_size_train = 13


df = pd.read_excel("path to df")
df_rot_mips_collages = pd.read_excel("path to rot mips collages")
include_angels = [90]
df_rot_mips_collages = df_rot_mips_collages[df_rot_mips_collages.angle.isin(include_angels)].reset_index(drop=True)

path_output = "path of output"
outcome = "age"
pre_trained_weights = False

for k in tqdm(range(k_fold)):
    if k >= 0:
        print(f"Cross validation for fold {k}")
        max_epochs = 200
        val_interval = 5
        best_metric = "best_metric_classification"
        best_metric_epoch = "best_metrix_epoch"
        metric_values = []
        metric_values_r_squared = []
        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = DenseNet121(spatial_dims=10, in_channels=10, out_channels=10, init_features=64, dropout_prob=0.25).to(device)
        
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

        os.mkdir("dir path", k)

        factor = round(df.shape[0]/k_fold)
        if k == (k_fold - 1):
            df_val = df[factor*k:].reset_index(drop=True)
        else:
            df_val = df[factor*k:factor*k+factor].reset_index(drop=True)
        df_train = df[~df.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

        df_train_new = df_rot_mips_collages[df_rot_mips_collages.scan_date.isin(df_train.scan_date)].reset_index(drop=True)
        df_val_new = df_rot_mips_collages[df_rot_mips_collages.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

        print("Number of patients in Training set: ", len(df_train))
        print("Number of patients in Valdation set: ", len(df_val))

        train_files, train_loader = prepare_data(args, df_train_new, batch_size_train, shuffle=True, label=outcome)

        train_loss = []
        for epoch in tqdm(range(max_epochs)):

            #Training
            epoch_loss, train_loss = train_regression(model, train_loader, optimizer, loss_function, device, train_loss)
            print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")

            #Validation
            if (epoch + 1) % val_interval == 0:
                metric_values, best_metric_new = validation_regression(args, k, epoch, optimizer, model, df_val_new, device, best_metric, metric_values, metric_values_r_squared, path_output, outcome, loss_function)

                best_metric = best_metric_new

            #Save and plot
            np.save(os.path.join(path_output, "CV_" + str(k) + "/MAE.npy"), metric_values)

            path_MAE = os.path.join(path_output, "CV_" + str(k), "epoch_vs_MAE.jpg")

            if len(metric_values) > 2:
                plot(metric_values, path_MAE, "MAE")
                #plot(metric_values_r_squared, path_r_squared, "R2")
