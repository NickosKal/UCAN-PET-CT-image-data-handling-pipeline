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
import itertools
from sklearn.preprocessing import normalize


def save_as_gz(vimg,path):  
    '''

    Function to save the 3d simpleitk objects to disk(deprecated)

    '''
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


def make_isotropic(
    image,
    interpolator=sitk.sitkLinear,
    spacing=None,
    default_value=0,
    standardize_axes=False,
):
    
    """
    Many file formats (e.g. jpg, png,...) expect the pixels to be isotropic, same
    spacing for all axes. Saving non-isotropic data in these formats will result in
    distorted images. This function makes an image isotropic via resampling, if needed.
    Args:
        image (SimpleITK.Image): Input image.
        interpolator: By default the function uses a linear interpolator. For
                      label images one should use the sitkNearestNeighbor interpolator
                      so as not to introduce non-existant labels.
        spacing (float): Desired spacing. If none given then use the smallest spacing from
                         the original image.
        default_value (image.GetPixelID): Desired pixel value for resampled points that fall
                                          outside the original image (e.g. HU value for air, -1000,
                                          when image is CT).
        standardize_axes (bool): If the original image axes were not the standard ones, i.e. non
                                 identity cosine matrix, we may want to resample it to have standard
                                 axes. To do that, set this paramter to True.
    Returns:
        SimpleITK.Image with isotropic spacing which occupies the same region in space as
        the input image.
    """

    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    if spacing is None:
        spacing = min(original_spacing)
    new_spacing = [spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    new_direction = image.GetDirection()
    new_origin = image.GetOrigin()
    # Only need to standardize axes if user requested and the original
    # axes were not standard.
    if standardize_axes and not np.array_equal(
        np.array(new_direction), np.identity(image.GetDimension()).ravel()
    ):
        new_direction = np.identity(image.GetDimension()).ravel()
        # Compute bounding box for the original, non standard axes image.
        boundary_points = []
        for boundary_index in list(
            itertools.product(*zip([0] * image.GetDimension(), image.GetSize()))
        ):
            boundary_points.append(image.TransformIndexToPhysicalPoint(boundary_index))
        max_coords = np.max(boundary_points, axis=0)
        min_coords = np.min(boundary_points, axis=0)
        new_origin = min_coords
        new_size = (((max_coords - min_coords) / spacing).round().astype(int)).tolist()
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        new_origin,
        new_spacing,
        new_direction,
        default_value,
        image.GetPixelID(),
    )


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

    # print('index: ', np.array(vol_img.GetSize())/2.0)
    # print('physical rotation center: ', rotation_center)
    # print('old size: ', vol_img.GetSize())
    # print('max bound , min bound: ',max_bounds, ' ', min_bounds)
    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]
    # print('new size: ', new_sz)
    #new_sz = vol_img.GetSize()
    pix_array=sitk.GetArrayFromImage(vol_img)
    maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
    # print(maxtensity,mintensity)
    if modality == 'CT':
        default_pix_val=20


    else: #modality == 'PET':
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
                                        interpolator=sitk.sitkLinear,
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

