import matplotlib.pyplot as plt
import os
import sys
from tabulate import tabulate
parent_dir = os.path.abspath('../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import utils

# reading main config file
config = utils.read_config()

system = 2 # 1 or 2
if system == 1:
    PATH = config["Source"]["paths"]["source_path_system_1"]
elif system == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    PATH = config["Source"]["paths"]["source_path_system_2"]
else:
    PATH = ""
    print("Invalid system")

# For regression
regression_metrics = utils.evaluate_best_models_all_folds(system=2, type="regression", category=None, experiment_number=3, folds_list=list(range(10)))
print(regression_metrics)

# For classification
sex_classification_metrics = utils.evaluate_best_models_all_folds(system=2, type="classification", category="Sex", experiment_number=2, folds_list=list(range(10)))
print(sex_classification_metrics)

diagnosis_classification_metrics = utils.evaluate_best_models_all_folds_metric_based(system=1, type="classification", category="Diagnosis", experiment_number=2, folds_list=list(range(10)))
print(diagnosis_classification_metrics)


