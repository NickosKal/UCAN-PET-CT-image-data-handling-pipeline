import yaml
import os
import sys

import SimpleITK as sitk
import numpy as np
import pandas as pd
from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))

def read_config():
    #print(dir_path)
    with open(dir_path + "/config.yaml","r") as file_object:
        config=yaml.load(file_object,Loader=yaml.SafeLoader)
        #print(config)
    return config

#read_config()
"""
config = read_config()
print('-------------')
print(config[1]['resampling'])

source_path = config[0]['paths']['source_path']
incomplete_folders_filename = config[0]['filenames']['incomplete_folders_filename']
final_selected_images_filename = config[0]['filenames']['final_selected_images_filename']
list_of_distorted_images_filename = config[0]['filenames']['list_of_distorted_images_filename']

incomplete_folders_path = os.path.join(source_path, incomplete_folders_filename)
print(incomplete_folders_path)

final_selected_folders = os.path.join(source_path, final_selected_images_filename)
print(final_selected_folders)

list_of_distorted_images = os.path.join(source_path, list_of_distorted_images_filename)
print(list_of_distorted_images)
"""

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


'''

Function to save the 3d simpleitk objects to disk(deprecated)

'''

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

def save_projections_as_png(image,img_name, invert = True):
    '''

    Function to save 2d simpleitk projection objects as uint8 png images

    '''

    writer = sitk.ImageFileWriter()
    #img=sitk.Extract(image, image.GetSize())
    writer.SetFileName(img_name)
    # img_write=  sitk.Cast(    
    #     sitk.IntensityWindowing(
    #         image, windowMinimum=float(sitk.GetArrayFromImage(image).min()), windowMaximum=clip_value, outputMinimum=0.0, outputMaximum=255
    #     ),
    #     image.GetPixelID(),
    #     )
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
    projection = {'sum': sitk.SumProjection,
                'mean':  sitk.MeanProjection,
                'std': sitk.StandardDeviationProjection,
                'min': sitk.MinimumProjection,
                'max': sitk.MaximumProjection}
    
    #vol_img = make_isotropic(vol_img)

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
        #rotation_transform.SetRotation(0,0,ang)    
        all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])
        
    all_points = np.array(all_points)
    min_bounds = all_points.min(0)
    max_bounds = all_points.max(0)


    #resampling grid will be isotropic so no matter which direction we project to
    #the images we save will always be isotropic (required for vol_img formats that 
    #assume isotropy - jpg,png,tiff...)

    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]
    # print('new size: ', new_sz)
    #new_sz = vol_img.GetSize()
    pix_array=sitk.GetArrayFromImage(vol_img)
    maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
    # print(maxtensity,mintensity)
    if modality == 'CT':
        default_pix_val=20


    elif modality == 'PET':
        default_pix_val=0
        #clipping intensities
        clamper = sitk.ClampImageFilter()
        clamper.SetLowerBound(0)
        clamper.SetUpperBound(clip_value)
        vol_img=clamper.Execute(vol_img)
        # vol_img = sitk.Cast(    
        # sitk.IntensityWindowing(
        #     vol_img, windowMinimum=mintensity, windowMaximum=clip_value, outputMinimum=0.0, outputMaximum=255
        # ),
        # vol_img.GetPixelID(),
        # )

    for ang in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, ang) 
        #rotation_transform.SetRotation(0,0,ang)
        resampled_image = sitk.Resample(image1=vol_img,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkNearestNeighbor,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection = vol_img.GetDirection(), #[1,0,0,0,1,0,0,0,1]
                                        defaultPixelValue = default_pix_val, 
                                        outputPixelType = vol_img.GetPixelID())
        if modality=='CT':
            masked_resampled_image=get_proj_after_mask(resampled_image,maxtensity,mintensity,t_type)
        else:
            masked_resampled_image=resampled_image

        proj_image = projection[ptype](masked_resampled_image, paxis)
        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0 
        axes_shifted_pi=sitk.Extract(proj_image, extract_size) #flip axes

        if save_img:
            imgname= img_n + r'_{0}_image_{1}'.format(modality + '_' + t_type,(180 * ang/np.pi) )
            save_projections_as_png(axes_shifted_pi, imgname + '.png', invert_intensity) #sitk.InvertIntensity(axes_shifted_pi,maximum=1)
            save_projections_as_nparr(axes_shifted_pi, imgname, invert_intensity)
    print(f'Finished generating {int(180.0/angle)+1} - {ptype} intensity 2D projections from the {modality} volume image! ')

# def get_2D_projections(vol_img,modality,ptype,angle,t_type='N',save_img=True,img_n=''):
#     projection = {'sum': sitk.SumProjection,
#                 'mean':  sitk.MeanProjection,
#                 'std': sitk.StandardDeviationProjection,
#                 'min': sitk.MinimumProjection,
#                 'max': sitk.MaximumProjection}
    
#     #vol_img = make_isotropic(vol_img)

#     paxis = 0
#     rotation_axis = [0,0,1]
#     rotation_angles = np.linspace(-np.pi/2, np.pi/2, int( (np.pi / (  ( angle / 180 ) * np.pi ) ) + 1 ) ) # angle range- [0, +180];
#     rotation_center = vol_img.TransformContinuousIndexToPhysicalPoint(np.array(vol_img.GetSize())/2.0) #[(index-1)/2.0 for index in vol_img.GetSize()])

#     rotation_transform = sitk.VersorRigid3DTransform()
#     #rotation_transform = sitk.Euler3DTransform()
#     rotation_transform.SetCenter(rotation_center)

#     #Compute bounding box of rotating volume and the resampling grid structure
#     image_indexes = list(zip([0,0,0], [sz-1 for sz in vol_img.GetSize()]))
#     image_bounds = []
#     for i in image_indexes[0]:
#         for j in image_indexes[1]:
#             for k in image_indexes[2]:
#                 image_bounds.append(vol_img.TransformIndexToPhysicalPoint([i,j,k]))

#     all_points = []
#     for ang in rotation_angles:
#         rotation_transform.SetRotation(rotation_axis, ang)    
#         #rotation_transform.SetRotation(0,0,ang)    
#         all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])
        
#     all_points = np.array(all_points)
#     min_bounds = all_points.min(0)
#     max_bounds = all_points.max(0)


#     #resampling grid will be isotropic so no matter which direction we project to
#     #the images we save will always be isotropic (required for vol_img formats that 
#     #assume isotropy - jpg,png,tiff...)

#     # print('index: ', np.array(vol_img.GetSize())/2.0)
#     # print('physical rotation center: ', rotation_center)
#     # print('old size: ', vol_img.GetSize())
#     # print('max bound , min bound: ',max_bounds, ' ', min_bounds)
#     new_spc = [np.min(vol_img.GetSpacing())]*3
#     new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]
#     # print('new size: ', new_sz)
#     #new_sz = vol_img.GetSize()
#     pix_array=sitk.GetArrayFromImage(vol_img)
#     maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
#     # print(maxtensity,mintensity)
#     if modality == 'CT':
#         default_pix_val=20


#     else:
#         default_pix_val=0
#         #clipping intensities
#         clamper = sitk.ClampImageFilter()
#         clamper.SetLowerBound(0)
#         clamper.SetUpperBound(15)
#         vol_img=clamper.Execute(vol_img)
#         # vol_img = sitk.Cast(    
#         # sitk.IntensityWindowing(
#         #     vol_img, windowMinimum=mintensity, windowMaximum=clip_value, outputMinimum=0.0, outputMaximum=255
#         # ),
#         # vol_img.GetPixelID(),
#         # )

#     for ang in rotation_angles:
#         rotation_transform.SetRotation(rotation_axis, ang) 
#         #rotation_transform.SetRotation(0,0,ang)
#         resampled_image = sitk.Resample(image1=vol_img,
#                                         size=new_sz,
#                                         transform=rotation_transform,
#                                         interpolator=sitk.sitkNearestNeighbor,
#                                         outputOrigin=min_bounds,
#                                         outputSpacing=new_spc,
#                                         outputDirection = vol_img.GetDirection(), #[1,0,0,0,1,0,0,0,1]
#                                         defaultPixelValue = default_pix_val, 
#                                         outputPixelType = vol_img.GetPixelID())
#         """
#         if modality=='CT':
#             masked_resampled_image=get_proj_after_mask(resampled_image,maxtensity,mintensity,t_type)
#         else:
#             masked_resampled_image=resampled_image
#         """

#         proj_image = projection[ptype](resampled_image, paxis)
#         extract_size = list(proj_image.GetSize())
#         extract_size[paxis]=0 
#         axes_shifted_pi=sitk.Extract(proj_image, extract_size) #flip axes

#         if save_img:
#             imgname= img_n + r'{0}'.format((180 * ang/np.pi) )
#             save_projections_as_png(axes_shifted_pi, imgname + '.png') #sitk.InvertIntensity(axes_shifted_pi,maximum=1)
#             save_projections_as_nparr(axes_shifted_pi, imgname)
#     print(f'Finished generating {int(180.0/angle)+1} - {ptype} intensity 2D projections from the {modality} volume image! ')

def compute_suv(vol_img, PatientWeight, AcquisitionTime , RadiopharmaceuticalStartTime, RadionuclideHalfLife, RadionuclideTotalDose):
    
    estimated = False

    raw = sitk.GetArrayFromImage(vol_img)    
    spacing = vol_img.GetSpacing()
    origin = vol_img.GetOrigin()
    direction = vol_img.GetDirection() 

    #raw,spacing,origin,direction = imread(image_file_list)
    
    
    try:
        weight_grams = float(PatientWeight)*1000
    except:
        #traceback.print_exc()
        weight_grams = 75000
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
        #traceback.print_exc()
        decay = np.exp(-np.log(2)*(1.75*3600)/6588); # 90 min waiting time, 15 min preparation
        injected_dose_decay = 420000000 * decay; # 420 MBq
        estimated = True
    
    # Calculate SUV # g/ml
    suv = raw*weight_grams/injected_dose_decay
    
    return suv, estimated, raw,spacing,origin,direction