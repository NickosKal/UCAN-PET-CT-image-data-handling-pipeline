import os
import shutil
import string
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
from monai.networks.nets.densenet import Densenet121
from tqdm import tqdm
from monai.transforms.post.array import Activations, AsDiscrete
from monai.transforms.utility.array import EnsureChannelFirst
from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.spatial.array import RandFlip,RandRotate,RandZoom
from monai.transforms.intensity.array import ScaleIntensity

from monai.utils.misc import set_determinism
from generate_dataset import prepare_data
import sys

parent_dir = os.path.abspath('../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Task4.utils import (plot_c_k_score, 
                         train_classification, 
                         validation_sex_classification,
                         validation_diagnosis_classification,
                         plot_auc, 
                         working_system)

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

def stratified_split(df_clean, k):
    df_list=[ df_clean[df_clean['GT_diagnosis_label']==x].reset_index(drop=True) for x in range(3) ]
    factor_list= [ round(x.shape[0]/k_fold) for x in df_list ]

    if k == (k_fold - 1):
        patients_for_val = []
        for x,f in zip(df_list,factor_list):
            patients_for_val.extend(x[f*k:].patient_ID.tolist())
        df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

    else:
        patients_for_val = []
        for x,f in zip(df_list,factor_list):
            patients_for_val.extend(x[f*k:f*k+f].patient_ID.tolist())
        df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

    df_train = df_clean[~df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

    return df_train, df_val

outcome = "sex" # GT_diagnosis_label
experiment = 1
k_fold = 10
learning_rate = 1e-4
weight_decay = 1e-5
batch_size_train = 10
args = {"num_workers": 2,
        "batch_size_val": 1}

df_path = PATH + config["collages_for_classification_dataframe"]
df = pd.read_excel(df_path)
df_sorted = df.sort_values(by="patient_ID")

path_output = PATH + config['classification_path']

path_output_for_sex = os.path.join(path_output, "Sex" + "/" + "Experiment_" + str(experiment) + "/")
if not os.path.exists(path_output_for_sex):
    os.makedirs(path_output_for_sex)

path_output_for_diagnosis = os.path.join(path_output, "Diagnosis" + "/" + "Experiment_" + str(experiment) + "/")
if not os.path.exists(path_output_for_diagnosis):
    os.makedirs(path_output_for_diagnosis)

if outcome == "sex":
    folder_name = "Sex"
    output_channels = 2
elif outcome == "GT_diagnosis_label":
    folder_name = "Diagnosis"
    output_channels = 3
else:
    folder_name = ""
    output_channels = 1

pre_trained_weights = False

for k in tqdm(range(k_fold)):

    if k >= 0:
        print("Cross Validation for fold: {}".format(k))
        max_epochs = 100
        val_interval = 1
        best_metric = 0
        best_metric_epoch = -1
        metric_values = []
        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model = Densenet121(spatial_dims=2, in_channels=10, out_channels=output_channels, init_features=64, dropout_prob=0.25).to(device)
        
        if pre_trained_weights:
            # Load pre trained weights
            print("Checkpoint Loading for Cross Validation: {}".format(k))
            checkpoint_path, epoch_to_continue = utils.load_checkpoints(system, "classification", outcome, experiment, k)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["net"])
        else:
            print("Training from Scratch!!")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        if outcome == "GT_diagnosis_label":
            if not os.path.exists(path_output_for_diagnosis + "CV_" + str(k) + '/Network_Weights/'):
                os.makedirs(path_output_for_diagnosis + "CV_" + str(k) + '/Network_Weights/')

            if not os.path.exists(path_output_for_diagnosis + "CV_" + str(k) + '/Metrics/'):
                os.makedirs(path_output_for_diagnosis + "CV_" + str(k) + '/Metrics/')
            
            if not os.path.exists(path_output_for_diagnosis + "CV_" + str(k) + '/MIPs/'):
                os.makedirs(path_output_for_diagnosis + "CV_" + str(k) + '/MIPs/')

            df_train, df_val = stratified_split(df_sorted, k)

            print("Number of exams in Training set: ", len(df_train))
            print("Number of exams in Validation set: ", len(df_val))

            class_freq_diagnosis = np.unique(df_train["GT_diagnosis_label"], return_counts=True)[1]
            class_weights_diagnosis = torch.tensor([float(class_freq_diagnosis[0]/np.sum(class_freq_diagnosis)), float(class_freq_diagnosis[1]/np.sum(class_freq_diagnosis)), float(class_freq_diagnosis[2]/np.sum(class_freq_diagnosis))]).to(device)
            print("class_weights_diagnosis: ", class_weights_diagnosis)
            loss_function_diagnosis = torch.nn.CrossEntropyLoss(weight=class_weights_diagnosis)

            train_files, train_loader = prepare_data(args, df_train, batch_size_train, shuffle=True, label=outcome)

            train_loss = []
            for epoch in tqdm(range(max_epochs)):
                
                if pre_trained_weights:
                    epoch = epoch + int(epoch_to_continue) + 1
                else:
                    pass

                epoch_loss, train_loss = train_classification(model, train_loader, optimizer, loss_function_diagnosis, device, train_loss, outcome)
                print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")

                if (epoch + 1) % val_interval == 0:
                    metric_values, best_metric_new = validation_diagnosis_classification(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, path_output_for_diagnosis, outcome)
                    best_metric = best_metric_new

                np.save(os.path.join(path_output_for_diagnosis, "CV_" + str(k) + "/c_k_score.npy"), metric_values)
                path_dice = os.path.join(path_output_for_diagnosis, "CV_" + str(k), "epoch_vs_c_k_score.jpg")

                if len(metric_values) > 2:
                    plot_c_k_score(metric_values, path_dice)

        elif outcome == "sex":
            if not os.path.exists(path_output_for_sex + "CV_" + str(k) + '/Network_Weights/'):
                os.makedirs(path_output_for_sex + "CV_" + str(k) + '/Network_Weights/')

            if not os.path.exists(path_output_for_sex + "CV_" + str(k) + '/Metrics/'):
                os.makedirs(path_output_for_sex + "CV_" + str(k) + '/Metrics/')
            
            if not os.path.exists(path_output_for_sex + "CV_" + str(k) + '/MIPs/'):
                os.makedirs(path_output_for_sex + "CV_" + str(k) + '/MIPs/')

            df_train, df_val = stratified_split(df_sorted, k)

            print("Number of exams in Training set: ", len(df_train))
            print("Number of exams in Validation set: ", len(df_val))

            print("Patient's sex distribution in Training set: ", df_train.groupby('sex')['patient_ID'].nunique())
            print("Patient's sex distribution in Validation set: ", df_val.groupby('sex')['patient_ID'].nunique())

            # Use this when training for sex classification
            class_freq_sex = np.unique(df_train["sex"], return_counts=True)[1]
            class_weights_sex = torch.tensor([float(class_freq_sex[0]/np.sum(class_freq_sex)), float(class_freq_sex[1]/np.sum(class_freq_sex))]).to(device)
            loss_function_sex = torch.nn.CrossEntropyLoss(weight=class_weights_sex)

            train_files, train_loader = prepare_data(args, df_train, batch_size_train, shuffle=True, label=outcome)

            train_loss = []
            for epoch in tqdm(range(max_epochs)):
                
                if pre_trained_weights:
                    epoch = epoch + int(epoch_to_continue) + 1
                else:
                    pass

                epoch_loss, train_loss = train_classification(model, train_loader, optimizer, loss_function_sex, device, train_loss, outcome)
                print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")

                if (epoch + 1) % val_interval == 0:
                    metric_values, best_metric_new = validation_sex_classification(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, path_output_for_sex, outcome)
                    best_metric = best_metric_new

                np.save(os.path.join(path_output_for_sex, "CV_" + str(k) + "/AUC.npy"), metric_values)
                path_dice = os.path.join(path_output_for_sex, "CV_" + str(k), "epoch_vs_AUC.jpg")

                if len(metric_values) > 2:
                    plot_auc(metric_values, path_dice)
