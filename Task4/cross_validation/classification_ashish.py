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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from generate_dataset import prepare_data
import sys

parent_dir = os.path.abspath('../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Task4.utils import train_classification, validation_classification, plot_auc

experiment = "4"
k_fold = 4
learning_rate = 1e-4
weight_decay = 1e-5
batch_size_train = 10
args = {"num_workers": 4,
        "batch_size_val": 1}

#df = pd.read_excel("/media/andres/T7 Shield1/UCAN_project/dataset_for_training_classification.xlsx")
df = pd.read_excel("/home/ashish/Ashish/UCAN/ReshapedCollages/dataset_for_training_classification_v2.xlsx")
print(df.shape)
df_sorted = df.sort_values(by="patient_ID")

try:
    df_clean = df_sorted.drop(columns="Unnamed: 0").reset_index(drop=True)
except:
    df_clean = df_sorted.copy()

#path_output = "/media/andres/T7 Shield1/UCAN_project/Results/classification"
path_output = "/home/ashish/Ashish/UCAN/Results/classification/experiment_" + experiment + "/"
outcome = "sex" # diagnosis

#checkpoint_path = "/home/ashish/Ashish/UCAN/pretrained_model_autoPet/classification_sex/best_model_10.pth.tar"
pre_trained_weights = False

for k in tqdm(range(k_fold)):
    if k >= 0:
        print("Cross Validation for fold: {}".format(k))
        max_epochs = 200
        val_interval = 1
        best_metric = 0
        best_metric_epoch = -1
        metric_values = []
        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Densenet121(spatial_dims=2, in_channels=10, out_channels=2, init_features=64, dropout_prob=0.25).to(device)
        
        if pre_trained_weights:
            # Load pre trained weights
            #print("Checkpoint Loading for Cross Validation: {}".format(k))
            # checkpoint_path = load_checkpoint(args, k)
            #checkpoint = torch.load(checkpoint_path)
            #model.load_state_dict(checkpoint["net"])
            pass
        else:
            print("Training from Scratch!!")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        if not os.path.exists(path_output+"CV_"+str(k)+'/Network_Weights/'):
            os.makedirs(path_output+"CV_"+str(k)+'/Network_Weights/')

        if not os.path.exists(path_output+"CV_"+str(k)+'/Metrics/'):
            os.makedirs(path_output+"CV_"+str(k)+'/Metrics/')
        
        if not os.path.exists(path_output+"CV_"+str(k)+'/MIPs/'):
            os.makedirs(path_output+"CV_"+str(k)+'/MIPs/')

        """
        os.mkdir("dir path", k)
        factor = round(df.shape[0]/k_fold)
        if k == (k_fold - 1):
            df_val = df[factor*k:].reset_index(drop=True)
        else:
            df_val = df[factor*k:factor*k+factor].reset_index(drop=True)
        
        df_train = df[~df.scan_date.isin(df_val.scan_date)].reset_index(drop=True)
        """

        factor = round(df.shape[0]/k_fold)
        if k == (k_fold - 1):
            patients_for_val = df_clean[factor*k:].patient_ID.tolist()
            df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)
        else:
            patients_for_val = df_clean[factor*k:factor*k+factor].patient_ID.tolist()
            df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

        df_train = df_clean[~df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)


        print("Number of exams in Training set: ", len(df_train))
        print("Number of patients in Training set: ", df_train.npr.nunique())
        print("Patient's sex distribution in Training set: ", df_train.groupby('sex')['npr'].nunique())
        print("Number of exams in Validation set: ", len(df_val))
        print("Number of patients in Validation set: ", df_val.npr.nunique())
        print("Patient's sex distribution in Validation set: ", df_val.groupby('sex')['npr'].nunique())

        class_freq = np.unique(df_train["sex"], return_counts=True)[1]
        class_weights = torch.tensor([float(class_freq[0]/np.sum(class_freq)), float(class_freq[1]/np.sum(class_freq))]).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        train_files, train_loader = prepare_data(args, df_train, batch_size_train, shuffle=True, label=outcome)

        train_loss = []
        for epoch in tqdm(range(max_epochs)):
            epoch_loss, train_loss = train_classification(model, train_loader, optimizer, loss_function, device, train_loss, outcome)
            print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                metric_values, best_metric_new = validation_classification(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, path_output, outcome)
                best_metric = best_metric_new

            np.save(os.path.join(path_output, "CV_" + str(k) + "/AUC.npy"), metric_values)
            path_dice = os.path.join(path_output, "CV_" + str(k), "epoch_vs_auc.jpg")
            if len(metric_values) > 2:
                plot_auc(metric_values, path_dice)
