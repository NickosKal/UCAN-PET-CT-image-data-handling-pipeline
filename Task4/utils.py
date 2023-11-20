from re import S
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from tabulate import tabulate
import torch
import torch.nn as nn
from monai.inferers.utils import sliding_window_inference
from monai.data.dataset import CacheDataset, Dataset, SmartCacheDataset
from monai.data.dataset import CacheDataset
from monai.data.utils import list_data_collate, decollate_batch
from monai.data.dataloader import DataLoader
from monai.data.image_dataset import ImageDataset
from tqdm import tqdm
import cc3d
import SimpleITK as sitk
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import nibabel as nib
import scipy.ndimage
from sklearn.metrics import mean_absolute_error, r2_score
from Task4.cross_validation.generate_dataset import prepare_data

def make_dirs(path, k):
    path_CV = os.path.join(path, "CV_" + str(k))
    if not os.path.exists(path_CV):
        os.mkdir(path_CV)
    path_Network_weights = os.path.join(path_CV, "Network_Weights")
    if not os.path.exists(path_Network_weights):
        os.mkdir(path_Network_weights)
    path_MIPs = os.path.join(path_CV, "MIPs")
    if not os.path.exists(path_MIPs):
        os.mkdir(path_MIPs)
    path_Metrics = os.path.join(path_CV, "Metrics")
    if not os.path.exists(path_Metrics):
        os.mkdir(path_Metrics)

def train_classification(model, train_loader, optimizer, loss_function, device, loss_values, outcome):
    m = nn.Softmax(dim=1)
    model.train()
    epoch_loss = 0
    step = 0
    for inputs, labels in tqdm(train_loader):

        if outcome == "diagnosis":            
            label_mapping = {'NEGATIVE': 0, 'LYMPHOMA': 1, 'LUNG_CANCER': 1, 'MELANOMA': 1}
            labels = [label_mapping[label] for label in labels]
            labels = torch.LongTensor(labels)
        else:
            labels = labels.type(torch.LongTensor)
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output_logits = model(inputs)
        #print(output_logits)
        #print(m(output_logits))
        #output_log_probs = torch.log_softmax(output_logits, dim=1)
        #print(m(outputs))
        loss = loss_function(output_logits, labels)
        print("loss: ", loss)
        #print("labels: ", labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    loss_values.append(epoch_loss)
    return epoch_loss, loss_values

def save_model(model, epoch, optimizer, k, path_Output):
    best_metric_epoch = epoch
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_metric_epoch}
    torch.save(state, os.path.join(path_Output, "CV_" + str(k) + "/Network_Weights/best_model_{}.pth.tar".format(best_metric_epoch)))

def validation_classification(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, path_Output, outcome):
    #df_performance = pd.DataFrame(columns=['pat_ID', 'scan_date', 'GT', 'prediction', 'prediction_probability (sex)'])
    df_performance = pd.DataFrame(columns=['pat_ID', 'scan_date', 'GT', 'prediction', 'prediction_probability (diagnosis)'])

    scan_dates = np.unique(df_val["scan_date"])
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    pred_prob = []
    GT = []
    metric_values = []

    for scan_date in tqdm(scan_dates):
        #Patient-wise Validation
        df_temp = df_val[df_val["scan_date"]==scan_date].reset_index(drop=True)
        pat_id = np.unique(df_temp["pat_ID"])
        val_files, val_loader = prepare_data(args, df_temp, args["batch_size_val"], shuffle=False, label=outcome)

        prediction_list = []
        pred_prob_female = []
        pred_prob_male = []
        for inputs, labels in val_loader:
            model.eval()

            if outcome == "diagnosis":            
                label_mapping = {'NEGATIVE': 0, 'LYMPHOMA': 1, 'LUNG_CANCER': 1, 'MELANOMA': 1}
                labels = [label_mapping[label] for label in labels]
                labels = torch.LongTensor(labels)

            #labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.numpy()
            #inputs = torch.unsqueeze(inputs, dim=0)
            outputs= torch.nn.Softmax(dim=1)(model(inputs))
            outputs = outputs.data.cpu().numpy()
            #print("outputs: ", outputs)
            if outputs[0][0] > outputs[0][1]:
                prediction = 0
            else:
                prediction = 1
            prediction_list.append(prediction)
            pred_prob_female.append(outputs[0][0])
            pred_prob_male.append(outputs[0][1])

        if np.mean(pred_prob_male) > np.mean(pred_prob_female):
            scan_prediction = 1
            #scan_pred_prob = np.mean(pred_prob_male)
        else:
            scan_prediction = 0
            #scan_pred_prob = np.mean(pred_prob_female)
        scan_pred_prob = np.mean(pred_prob_male)
        scan_GT = labels[0] # type: ignore

        #df_temp_new = pd.DataFrame({'pat_ID': [pat_id[0]], 'scan_date': [scan_date], 'GT': [scan_GT], 'prediction': [scan_prediction], 'prediction_probability (sex)': [scan_pred_prob]})
        df_temp_new = pd.DataFrame({'pat_ID': [pat_id[0]], 'scan_date': [scan_date], 'GT': [scan_GT], 'prediction': [scan_prediction], 'prediction_probability (diagnosis)': [scan_pred_prob]})

        df_performance = df_performance.append(df_temp_new, ignore_index=True) # type: ignore

        pred_prob.append(scan_pred_prob)
        GT.append(scan_GT)

    metric = calculate_metrics(pred_prob, np.array(GT).astype(int))
    print("AUC: ", metric)
    metric_values.append(metric)
    #Save the model if metric is increasing
    if metric > best_metric:
        best_metric = metric
        save_model(model, epoch, optimizer, k, path_Output)

    df_performance.to_csv(os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch) + ".csv"))
    return metric_values, best_metric

def calculate_metrics(pred_prob, GT):
    fpr, tpr, thresholds = metrics.roc_curve(GT, pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = (pred_prob >= optimal_threshold).astype(int)
    #print("prediction: ", pred_labels)
    #print("GT: ", GT)

    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    TP = ((pred_labels == 1) & (GT == 1)).sum()
    TN = ((pred_labels == 0) & (GT == 0)).sum()
    FP = ((pred_labels == 1) & (GT == 0)).sum()
    FN = ((pred_labels == 0) & (GT == 1)).sum()
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    specificity = TN / (TN + FP)
    auc = metrics.auc(fpr, tpr)

    results = [
        ["True Positives (TP)", TP],
        ["True Negatives (TN)", TN],
        ["False Positives (FP)", FP],
        ["False Negatives (FN)", FN],
        ["Sensitivity", sensitivity],
        ["Precision", precision],
        ["Specificity", specificity],
        ["AUC", auc]
    ]
    # Print results in tabular form
    print(tabulate(results, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    return auc

def huber_loss_function(predictions, targets, delta=5):
    errors = torch.abs(predictions - targets)
    quadratic_term = 0.5 * (errors ** 2)
    linear_term = delta * (errors - 0.5 * delta)
    loss = torch.where(errors < delta, quadratic_term, linear_term)
    return loss.mean()

def train_regression(model, train_files, train_loader, optimizer, loss_function, device, loss_values):
    model.train()
    epoch_loss = 0
    step = 0
    i = 0
    #print('train_regression')
    for inputs, labels in tqdm(train_loader):
        print(train_files[i]['SUV_MIP'])
        #labels = labels.type(torch.LongTensor)
        #print(inputs.shape)
        step += 1
        i += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        #print('1')
        output_logits = model(inputs).view(-1)
        #print('output_logits: ', output_logits)
        output_logits = output_logits.float()  # Convert logits to torch.float32
        labels = labels.float()
        #print("output: ", output_logits)
        #print("labels: ", labels)

        loss = loss_function(output_logits, labels)
        #loss = loss_function(predicted_output, labels)
        print('loss: ', loss)
        #loss = huber_loss_function(output_logits, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    #print("Train MSE: ", epoch_loss)
    loss_values.append(epoch_loss)
    return epoch_loss, loss_values

def validation_regression(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, metric_values_r_squared, path_Output, outcome, loss_function):
    df_performance = pd.DataFrame(columns=['patient_ID', 'scan_date', 'GT', 'prediction (age)'])
    #df_performance = pd.DataFrame(columns=['pat_ID', 'scan_date', 'GT', 'prediction (MTV (ml))'])
    #df_performance = pd.DataFrame(columns=['pat_ID', 'scan_date', 'GT', 'prediction (lean_volume (L))'])
    #df_performance = pd.DataFrame(columns=['pat_ID', 'scan_date', 'GT', 'lesion_count'])

    scan_dates = np.unique(df_val["scan_date"])
    prediction = []
    GT = []
    L1_loss = []
    MAE = []
    #metric_values
    for scan_date in tqdm(scan_dates):
        #Patient-wise Validation
        df_temp = df_val[df_val["scan_date"]==scan_date].reset_index(drop=True)
        pat_id = np.unique(df_temp["patient_ID"])
        #val_files, val_loader = prepare_data(args, df_temp, shuffle=False, label="age")
        val_files, val_loader = prepare_data(args, df_temp, args["batch_size_val"], shuffle=False, label=outcome)

        prediction_temp = []
        loss_temp = []
        #loss_temp_MAE = []
        for inputs, labels in val_loader:
            model.eval()
            #labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).view(-1)
            outputs = outputs.float()  # Convert logits to torch.float32
            labels = labels.float()
            #print(labels)
            loss = loss_function(outputs, labels)
            #loss = huber_loss_function(outputs, labels)

            #loss_MAE = loss_function(outputs, labels)

            outputs = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            #print("outputs: ", outputs)
            #print("labels: ", labels)

            prediction_temp.append(outputs)
            loss_temp.append(loss.data.cpu().numpy())

        scan_prediction = np.mean(prediction_temp)
        scan_GT = labels[0] # type: ignore
        scan_loss = np.mean(loss_temp)

        #print("GT: ", scan_GT)
        #print("Prediction: ", scan_prediction)

        df_temp_new = pd.DataFrame({'patient_ID': [pat_id[0]], 'scan_date': [scan_date], 'GT': [scan_GT], 'prediction (age)': [scan_prediction]})
        #df_temp_new = pd.DataFrame({'pat_ID': [pat_id[0]], 'scan_date': [scan_date], 'GT': [scan_GT], 'prediction (MTV (ml))': [scan_prediction]})
        #df_temp_new = pd.DataFrame({'pat_ID': [pat_id[0]], 'scan_date': [scan_date], 'GT': [scan_GT], 'prediction (lean_volume (L))': [scan_prediction]})
        #df_temp_new = pd.DataFrame({'pat_ID': [pat_id[0]], 'scan_date': [scan_date], 'GT': [scan_GT], 'lesion_count': [scan_prediction]})

        #df_performance = df_performance.append(df_temp_new, ignore_index=True) # type: ignore
        df_performance = pd.concat([df_performance, df_temp_new], ignore_index=True)

        prediction.append(scan_prediction)
        GT.append(scan_GT)
        L1_loss.append(scan_loss)

    #metric = np.mean(L1_loss)
    metric = mean_absolute_error(np.array(df_performance["GT"]), np.array(df_performance["prediction (age)"]))
    #metric = mean_absolute_error(np.array(df_performance["GT"]), np.array(df_performance["prediction (MTV (ml))"]))
    #metric = mean_absolute_error(np.array(df_performance["GT"]), np.array(df_performance["prediction (lean_volume (L))"]))
    #metric = mean_absolute_error(np.array(df_performance["GT"]), np.array(df_performance["lesion_count"]))

    #metric = r2_score(np.array(df_performance["GT"]), np.array(df_performance["lesion_count"]))


    ##metric_r_squared = r2_score(df_performance["GT"], df_performance["prediction (age)"])
    ##metric_r_squared = r2_score(df_performance["GT"]), np.array(df_performance["prediction (MTV (ml))"])
    ##metric_r_squared = r2_score(df_performance["GT"]), np.array(df_performance["prediction (lean_volume (L))"])

    #print("Validation Smooth L1 Loss: ", np.mean(L1_loss))
    print("Validation metric: ", metric)

    metric_values.append(metric)
    #metric_values_r_squared.append(metric_r_squared)

    #Save the model if metric is increasing
    if metric < best_metric:
        best_metric = metric
        save_model(model, epoch, optimizer, k, path_Output)

    df_performance.to_csv(os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch) + ".csv"))
    return metric_values, best_metric#, metric_values_r_squared


def plot_auc(dice, path):
    epoch = [1 * (i + 1) for i in range(len(dice))]
    plt.plot(epoch, dice)
    plt.savefig(path, dpi=400)
    plt.xlabel("Number of Epochs")
    plt.ylabel("AUC")

def plot(dice, path, name=None):
    epoch = [1 * (i + 1) for i in range(len(dice))]
    plt.plot(epoch, dice)
    plt.xlabel("Number of Epochs")
    plt.ylabel(name)
    plt.savefig(path, dpi=400)



