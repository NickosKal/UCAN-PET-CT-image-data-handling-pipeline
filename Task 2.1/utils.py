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

import SimpleITK as sitk
import numpy as np

def save_projections(image,img_name,max_intensity=np.double(50),min_intensity=np.double(-1024)):
    writer = sitk.ImageFileWriter()
    img=sitk.Extract(image, image.GetSize())
    writer.SetFileName(img_name)

    #img=make_isotropic(img)
    
    img_write=sitk.Cast(
        sitk.IntensityWindowing(
            img, windowMinimum=min_intensity, windowMaximum=max_intensity, outputMinimum=0.0, outputMaximum=100.0
        ),
        sitk.sitkUInt8,
    ) #outputMinimum=100.0, outputMaximum=255.0

    writer.Execute(img_write)  #sitk.Cast(sitk.RescaleIntensity(img,outputMinimum=0,outputMaximum=15)

def get_2D_projections(vol_img,ptype,angle,save_img=True,img_n='', modality=''):
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

    #resampling grid will be isotropic so no matter which direction we project to
    #the images we save will always be isotropic (required for vol_img formats that 
    #assume isotropy - jpg,png,tiff...)

    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]

    pix_array=sitk.GetArrayFromImage(vol_img)
    maxtensity,mintensity=pix_array.max(),pix_array.min()
    #print(maxtensity,mintensity)
    
    #initialize default HU unit based on modality
    if modality == 'CT':
        defHU = -20
    else:
        defHU = 0

    proj_images = []
    i=0
    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle) 
        resampled_image = sitk.Resample(image1=vol_img,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkLinear,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection = [1,0,0,0,1,0,0,0,1],
                                        defaultPixelValue =  defHU, #HU unit for air in CT, possibly set to 0 in other cases
                                        outputPixelType = vol_img.GetPixelID())
        proj_image = projection[ptype](resampled_image, paxis)
        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0
        proj_images.append(sitk.Extract(proj_image, extract_size))
        if save_img:
            img_name=img_n + r'_image_{0}.png'.format(i)
            save_projections(sitk.InvertIntensity(sitk.Extract(proj_image, extract_size), maximum=1),img_name=img_name, max_intensity=np.double(maxtensity), min_intensity=np.double(mintensity))
            #save_projections(sitk.Extract(proj_image, extract_size),img_name=img_name, max_intensity=np.double(maxtensity), min_intensity=np.double(mintensity))
            #save_projections(proj_image, img_name=img_name, max_intensity=np.double(maxtensity), min_intensity=np.double(mintensity))
            #print(i,angle)
        i+=1