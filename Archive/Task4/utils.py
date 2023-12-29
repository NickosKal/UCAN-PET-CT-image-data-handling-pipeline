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
import SimpleITK as sitk
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import scipy.ndimage
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, cohen_kappa_score
from torcheval.metrics.functional import multiclass_auroc, multiclass_accuracy, multiclass_recall, multiclass_precision
from Task4.cross_validation.generate_dataset import prepare_data
from Task4.cross_validation.generate_dataset_cpu import prepare_data

def working_system(system):
    if system == 1:
        pass        
    elif system == 2:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    else:
        print("Invalid system")

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
        #print("loss: ", loss)
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

def calculate_specificity(confusion_matrix_df, GT_class, pred_class):
    specificity = []
    for cls in GT_class:
        cls_orig = cls.split('_')[0]
        FP = np.sum(np.array(confusion_matrix_df.loc[[i for i in GT_class if i!=cls], [cls_orig + "_Pred"]]))
        TN = np.sum(np.array(confusion_matrix_df.loc[[i for i in GT_class if i!=cls], [i for i in pred_class if i!=cls_orig + "_Pred"]]))
        specificity.append(TN/(TN+FP))
    return specificity

def validation_diagnosis_classification(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, path_Output, outcome):
    df_performance = pd.DataFrame(columns=['patient_ID', 'scan_date', 'GT', 'prediction'])

    df_val["unique_pat_ID_scan_date"] = df_val.apply(lambda x: str(x["patient_ID"]) + "_" + str(x["scan_date"]), axis=1)
    unique_pat_ID_scan_date = np.unique(df_val["unique_pat_ID_scan_date"])
    pred_prob = []
    pred = []
    GT = []
    #metric_values = []

    for pat_ID_scan_date in tqdm(unique_pat_ID_scan_date):
        # Patient-wise Validation
        df_temp = df_val[df_val["unique_pat_ID_scan_date"] == pat_ID_scan_date].reset_index(drop=True)
        pat_id, scan_date = pat_ID_scan_date.split('_')
        val_files, val_loader = prepare_data(args, df_temp, args["batch_size_val"], shuffle=False, label=outcome)

        softmax_out = [0, 0, 0]
        softmax_prob_list = []
        for inputs, labels in val_loader:
            model.eval()

            labels = torch.LongTensor(labels)
            inputs, labels = inputs.to(device), labels.numpy()
            outputs= torch.nn.Softmax(dim=1)(model(inputs))
            
            softmax_out = outputs.data.cpu().numpy()
            softmax_out = softmax_out[0].tolist()
            softmax_prob_list.append(softmax_out)
            #print("outputs: ", outputs)

        scan_GT = labels[0] # type: ignore

        # For diagnosis classification
        df_temp_new = pd.DataFrame({'patient_ID': [pat_id], 
                                    'scan_date': [scan_date], 
                                    'GT': [scan_GT], 
                                    'prediction': [softmax_out.index(max(softmax_out))], 
                                    'prediction_probability C81(diagnosis)': [softmax_out[0]], 
                                    'prediction_probability C83 (diagnosis)': [softmax_out[1]], 
                                    'prediction_probability Others (diagnosis)': [softmax_out[2]]})
        
        df_performance = pd.concat([df_performance, df_temp_new], ignore_index=True)


        pred_prob.append(softmax_prob_list[0])
        pred.append(softmax_out.index(max(softmax_out)))
        GT.append(scan_GT)

    print("pred len: ", len(pred))
    print("GT len: ", len(GT))

    #diagnosis group      
    #idx_classes = ["C81_GT", "C83_GT", "Others_GT"]
    #col_classes = ["C81_Pred", "C83_Pred", "Others_Pred"]

    #new diagnosis
    idx_classes = ["C83.3_GT", "C81.1_GT", "C81.9_GT"]
    col_classes = ["C83.3_Pred", "C81.1_Pred", "C81.9_Pred"]

    confusion_matrix_df = pd.DataFrame(confusion_matrix(GT, pred), columns=col_classes, index=idx_classes)
    print(confusion_matrix_df)

    metric = calculate_multiclass_metrics(pred_prob, np.array(GT).astype(int), confusion_matrix_df)
    print("Cohen Kappa score: ", metric)

    metric_values.append(metric)
    #Save the model if metric is increasing
    if metric > best_metric:
        best_metric = metric
        save_model(model, epoch, optimizer, k, path_Output)

    df_performance.to_csv(os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch) + ".csv"), index=False)
    return metric_values, best_metric

def validation_sex_classification(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, path_Output, outcome):
    df_performance = pd.DataFrame(columns=['patient_ID', 'scan_date', 'GT', 'prediction'])

    df_val["unique_pat_ID_scan_date"] = df_val.apply(lambda x: str(x["patient_ID"]) + "_" + str(x["scan_date"]), axis=1)
    unique_pat_ID_scan_date = np.unique(df_val["unique_pat_ID_scan_date"])
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    pred_prob = []
    GT = []
    #metric_values = []

    for pat_ID_scan_date in tqdm(unique_pat_ID_scan_date):
        # Patient-wise Validation
        df_temp = df_val[df_val["unique_pat_ID_scan_date"] == pat_ID_scan_date].reset_index(drop=True)
        pat_id, scan_date = pat_ID_scan_date.split('_')
        val_files, val_loader = prepare_data(args, df_temp, args["batch_size_val"], shuffle=False, label=outcome)

        prediction_list = []
        pred_prob_female = []
        pred_prob_male = []
        for inputs, labels in val_loader:
            model.eval()

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
        else:
            scan_prediction = 0
        scan_pred_prob = np.mean(pred_prob_male)
        scan_GT = labels[0] 

        df_temp_new = pd.DataFrame({'patient_ID': [pat_id], 'scan_date': [scan_date], 'GT': [scan_GT], 'prediction': [scan_prediction],
                                        'prediction_probability (sex)': [scan_pred_prob]})

        #df_performance = df_performance.append(df_temp_new, ignore_index=True) # type: ignore
        df_performance = pd.concat([df_performance, df_temp_new], ignore_index=True)


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

def calculate_multiclass_metrics(pred_prob, GT, confusion_matrix_df):
    c_k_score = cohen_kappa_score(np.argmax(np.array(pred_prob),axis=1), GT)
    
    pred_prob = torch.tensor(pred_prob)
    GT = torch.tensor(GT)
    sensitivity = multiclass_recall(pred_prob, GT, average=None, num_classes=3) 
    precision = multiclass_precision(pred_prob, GT, average=None, num_classes=3)
    specificity = calculate_specificity(confusion_matrix_df, GT_class=list(confusion_matrix_df.index), pred_class=list(confusion_matrix_df.columns))#multiclass_accuracy(pred_prob, GT, average=None, num_classes=3)
    auc = multiclass_auroc(pred_prob, GT, num_classes=3)

    results = [
        ["Sensitivity", sensitivity],
        ["Precision", precision],
        ["Specificity", specificity],
        ["AUC", auc],
        ["Cohen Kappa Score", c_k_score]
    ]
    # Print results in tabular form
    print(tabulate(results, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    return c_k_score

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
    for inputs, labels in tqdm(train_loader):
        step += 1
        i += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output_logits = model(inputs).view(-1)
        output_logits = output_logits.float()  # Convert logits to torch.float32
        labels = labels.float()

        loss = loss_function(output_logits, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    loss_values.append(epoch_loss)
    return epoch_loss, loss_values

def validation_regression(args, k, epoch, optimizer, model, df_val, device, best_metric, metric_values, metric_values_r_squared, path_Output, outcome, loss_function):
    df_performance = pd.DataFrame(columns=['unique_patient_ID_scan_date', 'GT', 'prediction (age)'])
    #df_performance = pd.DataFrame(columns=['pat_ID', 'scan_date', 'GT', 'prediction (MTV (ml))'])
    #df_performance = pd.DataFrame(columns=['pat_ID', 'scan_date', 'GT', 'prediction (lean_volume (L))'])
    #df_performance = pd.DataFrame(columns=['pat_ID', 'scan_date', 'GT', 'lesion_count'])

    unique_patient_ID_scan_date = np.unique(df_val["unique_patient_ID_scan_date"])
    prediction = []
    GT = []
    L1_loss = []
    MAE = []
    #metric_values
    for unique_patient_ID_scan_date in tqdm(unique_patient_ID_scan_date):
        #Patient-wise Validation
        df_temp = df_val[df_val["unique_patient_ID_scan_date"]==unique_patient_ID_scan_date].reset_index(drop=True)
        val_files, val_loader = prepare_data(args, df_temp, args["batch_size_val"], shuffle=False, label=outcome)

        prediction_temp = []
        loss_temp = []
        #loss_temp_MAE = []
        for inputs, labels in val_loader:
            model.eval()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).view(-1)
            outputs = outputs.float()  # Convert logits to torch.float32
            labels = labels.float()
            loss = loss_function(outputs, labels)

            outputs = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            prediction_temp.append(outputs)
            loss_temp.append(loss.data.cpu().numpy())

        scan_prediction = np.mean(prediction_temp)
        scan_GT = labels[0] # type: ignore
        scan_loss = np.mean(loss_temp)

        df_temp_new = pd.DataFrame({'unique_patient_ID_scan_date': [unique_patient_ID_scan_date], 'GT': [scan_GT], 'prediction (age)': [scan_prediction]})

        df_performance = pd.concat([df_performance, df_temp_new], ignore_index=True)

        prediction.append(scan_prediction)
        GT.append(scan_GT)
        L1_loss.append(scan_loss)

    #metric = np.mean(L1_loss)
    metric = mean_absolute_error(np.array(df_performance["GT"]), np.array(df_performance["prediction (age)"]))

    #metric = r2_score(np.array(df_performance["GT"]), np.array(df_performance["lesion_count"]))


    ##metric_r_squared = r2_score(df_performance["GT"], df_performance["prediction (age)"])
    ##metric_r_squared = r2_score(df_performance["GT"]), np.array(df_performance["prediction (MTV (ml))"])
    ##metric_r_squared = r2_score(df_performance["GT"]), np.array(df_performance["prediction (lean_volume (L))"])

    print("Validation metric: ", metric)

    metric_values.append(metric)

    #Save the model if metric is increasing
    if metric < best_metric:
        best_metric = metric
        save_model(model, epoch, optimizer, k, path_Output)

    df_performance.to_csv(os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch) + ".csv"))
    return metric_values, best_metric#, metric_values_r_squared


def plot_c_k_score(dice, path):
    epoch = [1 * (i + 1) for i in range(len(dice))]
    plt.plot(epoch, dice)
    plt.savefig(path, dpi=400)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cohen Kappa Score")

def plot_auc(dice, path):
    epoch = [1 * (i + 1) for i in range(len(dice))]
    plt.plot(epoch, dice)
    plt.savefig(path, dpi=400)
    plt.xlabel("Number of Epochs")
    plt.ylabel("AUC")
    
def plot(dice, path, name="N"):
    epoch = [1 * (i + 1) for i in range(len(dice))]
    plt.plot(epoch, dice)
    plt.xlabel("Number of Epochs")
    plt.ylabel(name)
    plt.savefig(path, dpi=400)



