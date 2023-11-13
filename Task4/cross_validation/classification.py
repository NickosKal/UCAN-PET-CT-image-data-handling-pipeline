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
from utils import train_classification, validation_classification, plot_auc

k_fold = 15
learning_rate = 0
weight_decay = 0.001
batch_size_train = 13


df = pd.read_excel("dataset with suv ct paths and sex diagnosis and mtv")
df_rot_mips = pd.read_excel("dataset with mips")
df_rot_mips_collages = pd.read_excel("dataset with collages")

path_output = "where we want to store the outputs"
outcome = ["sex"] # what outcome we want to predict
pre_trained_weights = False

for k in tqdm(range(k_fold)):
    if k >= 0:
        print("Cross Validation for fold: {}".format(k))
        max_epochs = 200
        val_interval = 5
        best_metric = "best_metric_classification"
        best_metric_epoch = "best_metrix_epoch"
        metric_values = []
        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Densenet121(spatial_dims=10, in_channels=10, out_channels=10, init_features=64, dropout_prob=0.25).to(device)
        
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
        print("Number of patients in Validation set: ", len(df_val))

        class_freq = np.unique(df_train_new["sex"], return_counts=True)[1]
        class_weights = torch.tensor([float(class_freq[0]/np.sum(class_freq)), float(class_freq[1]/np.sum(class_freq))]).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        train_files, train_loader = prepare_data(args, df_train_new, batch_size_train, shuffle=True, label=outcome)

        train_loss = []
        for epoch in tqdm(range(max_epochs)):
            epoch_loss, train_loss = train_classification(model, train_loader, optimizer, loss_function, device, train_loss, outcome)
            print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                metric_values, best_metric_new = validation_classification(args, k, epoch, optimizer, model, df_val_new, device, best_metric, metric_values, path_output, outcome)
                best_metric = best_metric_new

            np.save(os.path.join(path_output, "CV_ " + str(k) + "/AUC.npy"), metric_values)
            path_dice = os.path.join(path_output, "CV_" + str(k), "epoch_vs_auc.jpg")
            if len(metric_values) > 2:
                plot_auc(metric_values, path_dice)
