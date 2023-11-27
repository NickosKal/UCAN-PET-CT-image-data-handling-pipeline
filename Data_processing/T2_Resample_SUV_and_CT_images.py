import os
import sys
import shutil
import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pylab as plt
import SimpleITK as sitk
from datetime import datetime

%env SITK_SHOW_COMMAND "/home/andres/Downloads/Slicer-5.4.0-linux-amd64/Slicer"
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
dicom.config.convert_wrong_length_to_UN = True

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import utils






