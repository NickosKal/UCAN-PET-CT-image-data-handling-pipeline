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
from monai.networks.nets.densenet import Densenet121
from tqdm import tqdm
from monai.transforms.post.array import Activations, AsDiscrete
from monai.transforms.utility.array import EnsureChannelFirst
from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.spatial.array import RandFlip,RandRotate,RandZoom
from monai.transforms.intensity.array import ScaleIntensity

from monai.utils.misc import set_determinism
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from generate_dataset import prepare_data
import sys

parent_dir = os.path.abspath('../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Task4.utils import train_classification, validation_classification, plot_auc

experiment = 1
k_fold = 10
learning_rate = 5e-5
weight_decay = 5e-5
batch_size_train = 10
args = {"num_workers": 2,
        "batch_size_val": 1}

df = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/dataset_for_model_classification_training.xlsx")
df_sorted = df.sort_values(by="patient_ID")

outcome = "GT_diagnosis_label" # sex
path_output = "/media/andres/T7 Shield1/UCAN_project/Results/classification/"

path_output_for_sex = os.path.join(path_output, "Sex" + "/" + "Experiment_" + str(experiment) + "/")
if not os.path.exists(path_output_for_sex):
    os.makedirs(path_output_for_sex)

path_output_for_diagnosis = os.path.join(path_output, "Diagnosis" + "/" + "Experiment_" + str(experiment) + "/")
if not os.path.exists(path_output_for_diagnosis):
    os.makedirs(path_output_for_diagnosis)

pre_trained_weights = False

for k in tqdm(range(k_fold)):
    if k >= 0:
        print("Cross Validation for fold: {}".format(k))
        max_epochs = 100000
        val_interval = 5
        best_metric = 0
        best_metric_epoch = -1
        metric_values = []
        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        print(device)
        model = Densenet121(spatial_dims=2, in_channels=10, out_channels=3, init_features=64, dropout_prob=0.25).to(device)
        
        if pre_trained_weights:
            # Load pre trained weights
            # print("Checkpoint Loading for Cross Validation: {}".format(k))
            # checkpoint_path = load_checkpoint(args, k)
            # checkpoint = torch.load(checkpoint_path)
            # model.load_state_dict(checkpoint["net"])
            pass
        else:
            print("Training from Scratch!!")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        if not os.path.exists(path_output_for_diagnosis+"CV_"+str(k)+'/Network_Weights/'):
            os.makedirs(path_output_for_diagnosis+"CV_"+str(k)+'/Network_Weights/')

        if not os.path.exists(path_output_for_diagnosis+"CV_"+str(k)+'/Metrics/'):
            os.makedirs(path_output_for_diagnosis+"CV_"+str(k)+'/Metrics/')
        
        if not os.path.exists(path_output_for_diagnosis+"CV_"+str(k)+'/MIPs/'):
            os.makedirs(path_output_for_diagnosis+"CV_"+str(k)+'/MIPs/')

        factor = round(df.shape[0]/k_fold)
        if k == (k_fold - 1):
            patients_for_val = df[factor*k:].patient_ID.tolist()
            df_val = df[df.patient_ID.isin(patients_for_val)].reset_index(drop=True)
        else:
            patients_for_val = df[factor*k:factor*k+factor].patient_ID.tolist()
            df_val = df[df.patient_ID.isin(patients_for_val)].reset_index(drop=True)
        
        df_train = df[~df.patient_ID.isin(patients_for_val)].reset_index(drop=True)

        print("Number of patients in Training set: ", len(df_train))
        print("Number of patients in Validation set: ", len(df_val))

        class_freq_diagnosis = np.unique(df_train["GT_diagnosis_label"], return_counts=True)[1]
        class_weights_diagnosis = torch.tensor([float(class_freq_diagnosis[0]/np.sum(class_freq_diagnosis)), float(class_freq_diagnosis[1]/np.sum(class_freq_diagnosis)), float(class_freq_diagnosis[2]/np.sum(class_freq_diagnosis))]).to(device)
        print("class_weights_diagnosis: ", class_weights_diagnosis)
        loss_function_diagnosis = torch.nn.CrossEntropyLoss(weight=class_weights_diagnosis)

        # Use this when training for sex classification
        class_freq_sex = np.unique(df_train["sex"], return_counts=True)[1]
        class_weights_sex = torch.tensor([float(class_freq_sex[0]/np.sum(class_freq_sex)), float(class_freq_sex[1]/np.sum(class_freq_sex))]).to(device)
        loss_function_sex = torch.nn.CrossEntropyLoss(weight=class_weights_sex)

        
        train_files, train_loader = prepare_data(args, df_train, batch_size_train, shuffle=True, label=outcome)

        train_loss = []
        for epoch in tqdm(range(max_epochs)):
            epoch_loss, train_loss = train_classification(model, train_loader, optimizer, loss_function_diagnosis, device, train_loss, outcome)
            print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                metric_values, best_metric_new = validation_classification(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, path_output_for_diagnosis, outcome)
                best_metric = best_metric_new
            
            np.save(os.path.join(path_output_for_diagnosis, "CV_" + str(k) + "/AUC.npy"), metric_values)
            path_dice = os.path.join(path_output_for_diagnosis, "CV_" + str(k), "epoch_vs_auc.jpg")
            if len(metric_values) > 2:
                plot_auc(metric_values, path_dice)
