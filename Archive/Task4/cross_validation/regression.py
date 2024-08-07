from gc import callbacks
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
torch.cuda.empty_cache()
from tqdm import tqdm
from generate_dataset import prepare_data

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

experiment = 2
k_fold = 10
learning_rate = 1e-4
weight_decay = 1e-5
batch_size_train = 14
args = {"num_workers": 2,
        "batch_size_val": 1} #25

df_path = PATH + config['collages_for_rergession_dataframe']
df = pd.read_excel(df_path)

path_output = PATH + config['regression_path']

outcome = "patient_age" # "mtv"
pre_trained_weights = True

df = df.sort_values(by="scan_date")
df['scan_date'] = df['scan_date'].astype(str)
df['unique_patient_ID_scan_date'] = df['patient_ID'] + '_' + df['scan_date']
df = df.drop(columns=['patient_ID', 'scan_date'])

df = df.sort_values(by="unique_patient_ID_scan_date")
output_path = os.path.join(path_output, "Experiment_" + str(experiment) + "/")

for k in tqdm(range(k_fold)):
    if k >= 3:
        if k == 7:
            print("Already re-trained!!")
            continue
        else:
            print(f"Cross validation for fold {k}")

            max_epochs = 500
            val_interval = 1 
            best_metric = 100000000000
            best_metric_epoch = -1
            metric_values = []
            metric_values_r_squared = []
            print("Network Initialization")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            model = DenseNet121(spatial_dims=2, in_channels=10, out_channels=1, dropout_prob=0.25).to(device)
            
            if pre_trained_weights:
                # Use it in case we have pre trained weights
                print("Checkpoint Loading for Cross Validation: {}".format(k))
                checkpoint_path, epoch_to_continue = utils.load_checkpoints(system, "regression", None, experiment, k)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['net'])
            else:
                print("Training from Scratch!") 

            loss_function = torch.nn.SmoothL1Loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            
            if not os.path.exists(output_path + "CV_" + str(k) + '/Network_Weights/'):
                os.makedirs(output_path + "CV_" + str(k) + '/Network_Weights/')

            if not os.path.exists(output_path + "CV_" + str(k) + '/Metrics/'):
                os.makedirs(output_path + "CV_" + str(k) + '/Metrics/')
            
            if not os.path.exists(output_path + "CV_" + str(k) + '/MIPs/'):
                os.makedirs(output_path + "CV_" + str(k) + '/MIPs/')

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
                if pre_trained_weights:
                    epoch = epoch + int(epoch_to_continue) + 1
                else:
                    pass
                #Training
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
