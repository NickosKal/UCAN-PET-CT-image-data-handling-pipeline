import yaml
import os
import sys

import SimpleITK as sitk
import numpy as np
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

def save_as_gz(vimg,path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(vimg)


'''

Function to save 2d simpleitk projection objects as uint8 png images

'''

def save_projections(image,modality,img_name,max_intensity=50,min_intensity=-1024):
    writer = sitk.ImageFileWriter()
    #img=sitk.Extract(image, image.GetSize())
    writer.SetFileName(img_name)
    if modality=="CT":
        out_min=0.0
        out_max=255.0
    else:
        out_min=0.0
        out_max=15.0
    #img=make_isotropic(img)

    img_write=sitk.Cast(
        # sitk.IntensityWindowing(
        #     img, windowMinimum=min_intensity, windowMaximum=max_intensity, outputMinimum=out_min, outputMaximum=out_max
        # ),
        sitk.RescaleIntensity(image), #,outputMinimum=out_min,outputMaximum=out_max
        sitk.sitkUInt8
    ) #outputMinimum=100.0, outputMaximum=255.0

    writer.Execute(img_write)  #sitk.Cast(sitk.RescaleIntensity(img,outputMinimum=0,outputMaximum=15)

'''

Function to get 3D masks for CT scans to segment out the exact tissue type.

'''

def get_proj_after_mask(img,max_i,min_i,hu_type):
    multiply= sitk.MultiplyImageFilter()
    if hu_type == 'Bone' or hu_type == 'bone' or hu_type == 'B':
        seg = sitk.BinaryThreshold(
        img, lowerThreshold=200, upperThreshold=max_i,insideValue=1, outsideValue=0
        )
        op_img= multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
        # path= img_n + r'_{0}_image.nii'.format(modality + '_' + type)
        # save_as_gz(op_img,path)
    elif hu_type == 'Lean Tissue' or hu_type == 'lean' or hu_type == 'LT':
        seg = sitk.BinaryThreshold(
        img, lowerThreshold=-29, upperThreshold=150, insideValue=1, outsideValue=0
        )
        op_img= multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))

    elif hu_type == 'Adipose' or hu_type == 'adipose' or hu_type == 'AT':
        seg = sitk.BinaryThreshold(
        img, lowerThreshold=-199, upperThreshold=-30, insideValue=1, outsideValue=0
        )
        op_img= multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
        
    elif hu_type == 'Air' or hu_type == 'A':
        seg = sitk.BinaryThreshold(
        img, lowerThreshold=min_i, upperThreshold=-191, insideValue=1, outsideValue=0
        )
        op_img= multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
    
    elif hu_type=='N':
        op_img=img
    
    return op_img


def get_2D_projections(vol_img,modality,ptype,angle,t_type='N',save_img=True,img_n=''):
    projection = {'sum': sitk.SumProjection,
                'mean':  sitk.MeanProjection,
                'std': sitk.StandardDeviationProjection,
                'min': sitk.MinimumProjection,
                'max': sitk.MaximumProjection}
    paxis = 0
    rotation_axis = [0,0,1]
    rotation_angles = np.linspace(-1/2*np.pi, 1/2*np.pi, int(180.0/angle)) #15.0 degree 
    rotation_center = vol_img.TransformContinuousIndexToPhysicalPoint([(index-1)/2.0 for index in vol_img.GetSize()])

    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_center)

    #Compute bounding box of rotating volume and the resampling grid structure
    image_indexes = list(zip([0,0,0], [sz-1 for sz in vol_img.GetSize()]))
    image_bounds = []
    for i in image_indexes[0]:
        for j in image_indexes[1]:
            for k in image_indexes[2]:
                image_bounds.append(vol_img.TransformIndexToPhysicalPoint([i,j,k]))

    all_points = []
    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle)    
        all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])
        
    all_points = np.array(all_points)
    min_bounds = all_points.min(0)
    max_bounds = all_points.max(0)
    
    #path= img_n + r'_{0}_image.nii'.format(modality + '_' + t_type)
    #save_as_gz(new_vol_img,path)       


    #resampling grid will be isotropic so no matter which direction we project to
    #the images we save will always be isotropic (required for vol_img formats that 
    #assume isotropy - jpg,png,tiff...)

    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]
    
    if modality == 'CT':
        default_pix_val=20


    elif modality == 'PET':
        default_pix_val=0

        #clipping intensities

        # intensity_fil=sitk.IntensityWindowingImageFilter()
        # intensity_fil.SetWindowMaximum=maxtensity
        # intensity_fil.SetWindowMinimum=mintensity
        # intensity_fil.SetOutputMaximum=50000
        # intensity_fil.SetOutputMinimum=0
        # vol_img= intensity_fil.Execute(vol_img)

        pix_array=sitk.GetArrayFromImage(vol_img)
        maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
        print(maxtensity,mintensity)
        vol_img = sitk.Cast(    
        sitk.IntensityWindowing(
            vol_img, windowMinimum=mintensity, windowMaximum=maxtensity, outputMinimum=0.0, outputMaximum=15.0
        ),
        vol_img.GetPixelID(),
        )

        # print('After clipping:')
        # pix_array=sitk.GetArrayFromImage(vol_img)
        # maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
        # print(maxtensity,mintensity)

    elif modality == 'SUV':
        default_pix_val=0

        #clipping intensities

        # intensity_fil=sitk.IntensityWindowingImageFilter()
        # intensity_fil.SetWindowMaximum=maxtensity
        # intensity_fil.SetWindowMinimum=mintensity
        # intensity_fil.SetOutputMaximum=50000
        # intensity_fil.SetOutputMinimum=0
        # vol_img= intensity_fil.Execute(vol_img)

        pix_array=sitk.GetArrayFromImage(vol_img)
        maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
        print(maxtensity,mintensity)
        vol_img = sitk.Cast(    
        sitk.IntensityWindowing(
            vol_img, windowMinimum=mintensity, windowMaximum=maxtensity, outputMinimum=0.0, outputMaximum=2.0
        ),
        vol_img.GetPixelID(),
        )

        # print('After clipping:')
        # pix_array=sitk.GetArrayFromImage(vol_img)
        # maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
        # print(maxtensity,mintensity)


    proj_images = []
    i=0

    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle) 
        resampled_image = sitk.Resample(image1=vol_img,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkNearestNeighbor,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection = [1,0,0,0,1,0,0,0,1],
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
        proj_images.append(axes_shifted_pi) #sitk.Extract(proj_image, extract_size)

        #print('Size before saving: ',sitk.Extract(proj_image, extract_size).GetSize())
        
        if save_img:
            imgname= img_n + r'_{0}_image_{1}.png'.format(modality + '_' + t_type,i)
            save_projections(sitk.InvertIntensity(axes_shifted_pi,maximum=1), modality, imgname, max_intensity=maxtensity, min_intensity=mintensity)
            #save_projections(axes_shifted_pi, modality, imgname, max_intensity=maxtensity, min_intensity=mintensity)
        i+=1

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