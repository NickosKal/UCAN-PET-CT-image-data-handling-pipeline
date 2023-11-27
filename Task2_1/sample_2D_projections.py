from tkinter import Image
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def read_dicom(path: str):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    vol_img = reader.Execute()
    return vol_img

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


def get_proj_after_mask(img,max_i,min_i,hu_type):
    '''

    Function to get 3D masks for CT scans to segment out specific HU values.

    '''
    multiply= sitk.MultiplyImageFilter()
    if hu_type == 'Bone' or hu_type == 'bone' or hu_type == 'B':
        seg = sitk.BinaryThreshold(
        img, lowerThreshold=200, upperThreshold=max_i,insideValue=1, outsideValue=0
        )
        op_img= multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))

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
    
    else:
        op_img=img
    
    return op_img

def get_2D_projections(vol_img,modality,ptype,angle,invert_intensity = True, clip_value=15.0, t_type='N',save_img=True,img_n=''):
    ''' 
    Main function to get the 2D projections. \n
    \n
    Invert intensity- Bool variable to toggle while saving for intensity inversion\n
    Clip value- Intensity clipping, default is set to 15.0 \n
    Projection types: \n
                {'sum': sitk.SumProjection, \n
                'mean':  sitk.MeanProjection, \n
                'std': sitk.StandardDeviationProjection, \n
                'min': sitk.MinimumProjection, \n
                'max': sitk.MaximumProjection} \n
    Segmentation type:     \n
        Bone - 'Bone' or 'bone' or 'B' \n
        Lean Tissue- 'Lean Tissue' or 'lean' or 'LT' \n
        Adipose- 'Adipose' or 'adipose' or 'AT' \n
        Air- 'Air' or 'A' \n
        Anything else returns the projection without any masks\n
    '''
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
    pix_array=sitk.GetArrayFromImage(vol_img)
    maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())

    if modality == 'CT':
        default_pix_val=20


    else: #modality == 'PET':
        default_pix_val=0
        #clipping intensities
        clamper = sitk.ClampImageFilter()
        clamper.SetLowerBound(0)
        clamper.SetUpperBound(clip_value)
        vol_img=clamper.Execute(vol_img)

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



if __name__=="__main__":
    
    #for example - r'\..\_SUV_CT\201XXXX\CT.nii.gz'
    DICOM_PATH = '' 

    # Testing for CT
    # MODA='CT'
    # T_TYPE='B' 
    # PTYPE='mean'

    # Testing for PET
    MODA='PET' 
    T_TYPE='A' 
    PTYPE='max' 

    #for example - r'\..\2dprojections\20171207'
    # The function adds .png at the end for each angular projection image/np arr
    save_path=r''
    
    volimg=read_dicom(DICOM_PATH)

    get_2D_projections(volimg,MODA,ptype=PTYPE,angle=45,clip_value=15.0,img_n=save_path)


