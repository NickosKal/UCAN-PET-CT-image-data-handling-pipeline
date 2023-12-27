import yaml
import os
import sys
from PIL import Image
import SimpleITK as sitk
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import regex as re

dir_path = os.path.dirname(os.path.realpath(__file__))

def read_config():
    #print(dir_path)
    with open(dir_path + "/config.yaml","r") as file_object:
        config=yaml.load(file_object,Loader=yaml.SafeLoader)
        #print(config)
    return config

config = read_config()

def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)

def read_dicom(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    vol_img = reader.Execute()
    return vol_img

def save_as_gz(vimg,path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(vimg)

def create_collage(path_c, path_s, save_path):
    '''

    Function to save simple collage from 0.0 and -90.0 degree png images

    '''

    image1 = Image.open(path_c)
    image2 = Image.open(path_s)
    collage = Image.new('RGB', (image1.size[0]*2, image1.size[1]))
    # Paste the first image onto the collage
    collage.paste(image1, (0, 0))
    # Paste the second image onto the collage
    collage.paste(image2, (image1.size[0], 0))

    # Save the collage
    collage.save(save_path)

def save_projections_as_png(image,img_name, invert = True):
    '''

    Function to save 2d simpleitk projection objects as uint8 png images

    '''

    writer = sitk.ImageFileWriter()
    writer.SetFileName(img_name)
    img_write= sitk.Flip(image, [False, True]) #flipping across y axis
    img_write=sitk.Cast(
        sitk.RescaleIntensity(img_write), #sitk.RescaleIntensity()
        sitk.sitkUInt8
    )
    if invert:
        img_write   = sitk.InvertIntensity(img_write,maximum=255)
    else:
        pass
    writer.Execute(img_write)  #sitk.Cast(sitk.RescaleIntensity(img,outputMinimum=0,outputMaximum=15)

def save_projections_as_nparr(image,img_name, invert = True):
    '''

    Function to save 2d simpleitk projection images as a numpy array

    '''

    img = sitk.Flip(image, [False, True])

    if invert:
        img= sitk.InvertIntensity(img,maximum=255)
    else:
        pass
    
    arr= sitk.GetArrayFromImage(img)

    # Perform min-max normalization

    minv,maxv= np.min(arr), np.max(arr)
    arr_normed = (arr - minv) / (maxv - minv)
    np.save(img_name,np.array(arr_normed))

def get_proj_after_mask(img):
    
    '''

    Function to get 3D masks for CT scans to segment out the exact tissue type.

    '''

    pix_array=sitk.GetArrayFromImage(img)
    max_i, min_i=float(pix_array.max()),float(pix_array.min())

    #multiply= sitk.MultiplyImageFilter()
    #if hu_type == 'Bone' or hu_type == 'bone' or hu_type == 'B':
    bone_mask = sitk.BinaryThreshold(
    img, lowerThreshold=200, upperThreshold=max_i,insideValue=1, outsideValue=0
    )
    
    #bone_mask = multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
        # path= img_n + r'_{0}_image.nii'.format(modality + '_' + type)
        # save_as_gz(op_img,path)
    #elif hu_type == 'Lean Tissue' or hu_type == 'lean' or hu_type == 'LT':
    lean_mask = sitk.BinaryThreshold(
    img, lowerThreshold=-29, upperThreshold=150, insideValue=1, outsideValue=0
    )
    #lean_mask = multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))

    #elif hu_type == 'Adipose' or hu_type == 'adipose' or hu_type == 'AT':
    adipose_mask = sitk.BinaryThreshold(
    img, lowerThreshold=-199, upperThreshold=-30, insideValue=1, outsideValue=0
    )
    #adipose_mask = multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
        
    #elif hu_type == 'Air' or hu_type == 'A':
    air_mask = sitk.BinaryThreshold(
    img, lowerThreshold=min_i, upperThreshold=-191, insideValue=1, outsideValue=0
    )
    #air_mask = multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
    
    return bone_mask, lean_mask, adipose_mask, air_mask

def get_2D_projections(vol_img,modality,ptype,angle,invert_intensity = True, clip_value=15.0, t_type='N',save_img=True,img_n=''):
    """
    Input: PET/CT, Proj type, angle, min & max angle, HU Values/ mention: Bone/Adipose/Tissue

    Output: 2D sampled projections


    Notes:
    For Age,

    CT => lean sum of ,musles pixel, mean hu musle
    from musle take max PET  MIP

    CT, PT seperate

    CT(Grayscale) + PT(Color) = Overlap 2D img

    """

    projection = {'sum': sitk.SumProjection,
                'mean':  sitk.MeanProjection,
                'std': sitk.StandardDeviationProjection,
                'min': sitk.MinimumProjection,
                'max': sitk.MaximumProjection}
    
    paxis = 0
    rotation_axis = [0,0,1]
    rotation_angles = np.linspace(-np.pi/2, np.pi/2, int( (np.pi / (  ( angle / 180 ) * np.pi ) ) + 1 ) ) # angle range- [0, +180];
    rotation_center = vol_img.TransformContinuousIndexToPhysicalPoint(np.array(vol_img.GetSize())/2.0) #[(index-1)/2.0 for index in vol_img.GetSize()])

    rotation_transform = sitk.VersorRigid3DTransform()
    #rotation_transform = sitk.Euler3DTransform()
    rotation_transform.SetCenter(rotation_center)

    #Compute bounding box of rotating volume and the resampling grid structure
    image_indexes = list(zip([0,0,0], [sz-1 for sz in vol_img.GetSize()]))
    image_bounds = []
    for i in image_indexes[0]:
        for j in image_indexes[1]:
            for k in image_indexes[2]:
                image_bounds.append(vol_img.TransformIndexToPhysicalPoint([i,j,k]))

    all_points = []
    for ang in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, ang)     
        all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])
        
    all_points = np.array(all_points)
    min_bounds = all_points.min(0)
    max_bounds = all_points.max(0)

    #resampling grid will be isotropic so no matter which direction we project to
    #the images we save will always be isotropic (required for vol_img formats that 
    #assume isotropy - jpg,png,tiff...)

    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]
    pix_array=sitk.GetArrayFromImage(vol_img)
    maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
    
    """Setting a default pixel value based on modality (the resample function requires this argument as during rotation, 
                                      the pixel intensities for new locations are set to a default value) """
    if modality == 'CT':
        default_pix_val=20

    else:
        #Clipping intensities
        default_pix_val=0
        clamper = sitk.ClampImageFilter()
        clamper.SetLowerBound(0)
        clamper.SetUpperBound(clip_value)
        vol_img=clamper.Execute(vol_img)

    
    for ang in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, ang) 
        
        #Generate 3d volumes which are rotated by 'ang' angles
        resampled_image = sitk.Resample(image1=vol_img,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkLinear,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection = vol_img.GetDirection(), #[1,0,0,0,1,0,0,0,1]
                                        defaultPixelValue = default_pix_val, 
                                        outputPixelType = vol_img.GetPixelID())
        
        #Generate 2d projections from the rotated volume
        proj_image = projection[ptype](resampled_image, paxis)
        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0 
        axes_shifted_pi=sitk.Extract(proj_image, extract_size) #flip axes

        if save_img:
            #Save the projections as image or np array
            imgname= img_n + r'{0}'.format((180 * ang/np.pi))
            save_projections_as_png(axes_shifted_pi, imgname + '.png', invert_intensity)
            save_projections_as_nparr(axes_shifted_pi, imgname, invert_intensity)
    print(f'Finished generating {int(180.0/angle)+1} - {ptype} intensity 2D projections from the {modality} volume image! ')

def compute_suv(vol_img, PatientWeight, AcquisitionTime , RadiopharmaceuticalStartTime, RadionuclideHalfLife, RadionuclideTotalDose):
    
    estimated = False

    raw = sitk.GetArrayFromImage(vol_img)    
    spacing = vol_img.GetSpacing()
    origin = vol_img.GetOrigin()
    direction = vol_img.GetDirection() 
    
    try:
        weight_grams = float(PatientWeight)*1000
    except:
        weight_grams = 76274 #average weight from master data
        estimated = True
        
    try:
        # Get Scan time
        scantime = datetime.strptime(AcquisitionTime,'%H%M%S')
        # Start Time for the Radiopharmaceutical Injection
        injection_time = datetime.strptime(RadiopharmaceuticalStartTime,'%H%M%S')
        # Half Life for Radionuclide # seconds
        half_life = float(RadionuclideHalfLife) 
        # Total dose injected for Radionuclide
        injected_dose = float(RadionuclideTotalDose)
        # Calculate decay
        decay = np.exp(-np.log(2)*((scantime-injection_time).seconds)/half_life)
        # Calculate the dose decayed during procedure
        injected_dose_decay = injected_dose*decay; # in Bq        
    except:
        decay = 0.61440 #average decay in metadata
        injected_dose = 265987763 #average injected dose in metadata #265 MBq
        injected_dose_decay = injected_dose * decay; 
        estimated = True
    
    # Calculate SUV # g/ml
    suv = raw*weight_grams/injected_dose_decay
    
    return suv, estimated, raw,spacing,origin,direction

def find_distorted_examinations(path_of_exams, path_to_save):
    directory_list = list()
    for root, dirs, files in os.walk(path_of_exams, topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))
    
    dataset = pd.DataFrame(directory_list, columns=['directory'])
    countfiles_selected = {"directory": [], "count":[]}

    for index, row in dataset.iterrows():
        count = 0
        for path in os.listdir(row["directory"]):
            if os.path.isfile(os.path.join(row["directory"], path)):
                count += 1
                
        countfiles_selected["directory"].append(row["directory"])
        countfiles_selected["count"].append(count)

    countfiles_selected_df = pd.DataFrame.from_dict(countfiles_selected)
    exams_with_distorted_images_file = countfiles_selected_df[countfiles_selected_df["count"] < 179].reset_index()
    exams_with_distorted_images_file[['source_directory', 'patient_directory', 'PET-CT_info']] = exams_with_distorted_images_file['directory'].str.rsplit(pat='/', n=2, expand=True)
    exams_with_distorted_images_file.to_excel(path_to_save + "exams_with_distorted_images_file.xlsx")
    dataset = dataset[~dataset.directory.isin(exams_with_distorted_images_file.directory)]
    dataset.to_excel(path_to_save + "data_ready_for_filtering.xlsx")


def natural_sortkey(string):          
    tokenize = re.compile(r'(\d+)|(\D+)').findall
    return tuple(int(num) if num else alpha for num, alpha in tokenize(string.name))

def best_model_selection_from_fold(system, type, category, experiment_number, fold_number):
    if type == "regression":
        path = config["Source"]["paths"][f"source_path_system_{system}"] + config["regression_path"] + f"/Experiment_{experiment_number}/CV_{fold_number}/Network_Weights/"
    else:
        path = config["Source"]["paths"][f"source_path_system_{system}"] + config["classification_path"] + f"/{category}/Experiment_{experiment_number}/CV_{fold_number}/Network_Weights/"
    path_object = Path(path)
    models = path_object.glob("*")
    models_sorted = sorted(models, key=natural_sortkey)
    best_model_path = path + [model.name for model in models_sorted][-1]
    epoch_to_continue = best_model_path.split("_")[-1].split(".")[0]
    return best_model_path, epoch_to_continue
    
def load_checkpoints(system, type, category, experiment_number, fold_number):

    if fold_number == 0:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 1:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 2:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 3:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 4:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 5:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 6:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 7:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 8:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 9:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    
    return checkpoint_path, epoch_to_continue
