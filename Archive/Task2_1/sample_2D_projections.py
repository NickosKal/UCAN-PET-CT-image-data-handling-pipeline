from tkinter import Image
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt



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


def get_2D_projections(vol_img,modality,ptype,angle,invert_intensity = True,min_clip_value=0, max_clip_value=15.0,save_png=True,img_n=''):
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
    '''
    projection = {'sum': sitk.SumProjection,
                'mean':  sitk.MeanProjection,
                'std': sitk.StandardDeviationProjection,
                'min': sitk.MinimumProjection,
                'max': sitk.MaximumProjection}
    
    #vol_img = make_isotropic(vol_img)

    paxis = 0
    rotation_axis = [0,0,1]
    rotation_angles=[]
    rotation_angles.append(angle)
    #rotation_angles = np.linspace(-np.pi/2, np.pi/2, int( (np.pi / (  ( angle / 180 ) * np.pi ) ) + 1 ) ) # angle range- [0, +180];
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
        clamper.SetLowerBound(min_clip_value)
        clamper.SetUpperBound(max_clip_value)
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

        proj_image = projection[ptype](resampled_image, paxis)
        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0 
        axes_shifted_pi=sitk.Extract(proj_image, extract_size) #flip axes

        imgname= img_n + r'_{0}_image_{1}'.format(modality,(180 * ang/np.pi) )
        save_projections_as_nparr(axes_shifted_pi, imgname, invert_intensity)
        if save_png:
            save_projections_as_png(axes_shifted_pi, imgname + '.png', invert_intensity) #sitk.InvertIntensity(axes_shifted_pi,maximum=1)
            
    print(f'Finished generating {int(180.0/angle)+1} - {ptype} intensity 2D projections from the {modality} volume image! ')



if __name__=="__main__":
    
    #for example - r'\..\_SUV_CT\201XXXX\CT.nii.gz'
    VOL_IMG_PATH = r'' 

    # Testing for CT
    # MODA='CT'
    # PTYPE='mean'

    # Testing for PET
    MODA='PET' 
    PTYPE='max' 

    #for example - r'\..\2dprojections\20171207'
    # The function adds .png at the end for each angular projection image/np arr
    save_path=r''

    #Reading the 3D file first
    volimg=sitk.ReadImage(VOL_IMG_PATH)
    get_2D_projections(volimg,MODA,ptype=PTYPE,angle=45,clip_value=15.0,img_n=save_path)


