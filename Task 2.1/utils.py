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

def save_as_gz(vimg,path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(vimg)

def save_projections(image,modality,img_name,max_intensity=50,min_intensity=-1024):
    writer = sitk.ImageFileWriter()
    img=sitk.Extract(image, image.GetSize())
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
        sitk.RescaleIntensity(img,outputMinimum=out_min,outputMaximum=out_max),
        sitk.sitkUInt8
    ) #outputMinimum=100.0, outputMaximum=255.0

    writer.Execute(img_write)  #sitk.Cast(sitk.RescaleIntensity(img,outputMinimum=0,outputMaximum=15)

def get_proj_after_mask(img,hu_type):
    multiply= sitk.MultiplyImageFilter()
    if hu_type == 'Bone' or hu_type == 'bone' or hu_type == 'B':
        op_img= multiply.Execute(img,sitk.Cast(img > 200,img.GetPixelID()))
        # path= img_n + r'_{0}_image.nii'.format(modality + '_' + type)
        # save_as_gz(op_img,path)
    elif hu_type == 'Lean Tissue' or hu_type == 'lean' or hu_type == 'LT':
        print(hu_type)
        seg = sitk.BinaryThreshold(
        img, lowerThreshold=-29, upperThreshold=150, insideValue=1, outsideValue=0
        )
        #op_img= sitk.Cast(sitk.LabelOverlay(img, (img < 150 and img > -29) ),img.GetPixelID())
        #op_img= multiply.Execute(img,sitk.Cast(img < 150 and img > -29, img.GetPixelID()))
        op_img= multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
        print(op_img.GetSize(),seg.GetSize())

    elif hu_type == 'Adipose' or hu_type == 'adipose' or hu_type == 'AT':
        #op_img= sitk.Cast(img < -30 and img > -190,img.GetPixelID())
        op_img= multiply.Execute(img,(img < -30 and img > -190))
    elif hu_type == 'Air' or hu_type == 'A':
        #op_img= sitk.Cast(img < -191,img.GetPixelID())
        op_img= multiply.Execute(img,(img < -191))
    elif hu_type=='N':
        op_img=img
    return op_img


def get_2D_projections(vol_img,modality,t_type,ptype,angle,save_img=True,img_n=''):
    projection = {'sum': sitk.SumProjection,
                'mean':  sitk.MeanProjection,
                'std': sitk.StandardDeviationProjection,
                'min': sitk.MinimumProjection,
                'max': sitk.MaximumProjection}
    paxis = 0
    print(vol_img.GetSize())
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
    
    if modality == 'CT':
        default_pix_val=20


    elif modality == 'PET':
        default_pix_val=-50
        pass
    
    #path= img_n + r'_{0}_image.nii'.format(modality + '_' + t_type)
    #save_as_gz(new_vol_img,path)    


    #resampling grid will be isotropic so no matter which direction we project to
    #the images we save will always be isotropic (required for vol_img formats that 
    #assume isotropy - jpg,png,tiff...)

    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]
    print(new_spc)
    print(new_sz)
    pix_array=sitk.GetArrayFromImage(vol_img)
    maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
    #print(maxtensity,mintensity)
    
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
        proj_image = get_proj_after_mask(projection[ptype](resampled_image, paxis), t_type) 
        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0
        proj_images.append(proj_image) #sitk.Extract(proj_image, extract_size)
        if save_img:
            imgname= img_n + r'_{0}_image_{1}.png'.format(modality + '_' + t_type,i)
            save_projections(sitk.InvertIntensity(proj_image,maximum=1), modality, imgname, max_intensity=maxtensity, min_intensity=mintensity)

            #save_projections(sitk.Extract(proj_image, extract_size),img_name=img_name, max_intensity=np.double(maxtensity), min_intensity=np.double(mintensity))
            #save_projections(proj_image, img_name=img_name, max_intensity=np.double(maxtensity), min_intensity=np.double(mintensity))
            #print(i,angle)
        i+=1

#Testing
# if __name__=="__main__":
    # v_img = sitk.ReadImage(r"C:\Users\Audit\Uppsala - Masters Europe\Semester 3\Project in Image analysis - Software\UCAN-PET-CT-image-data-handling-pipeline\Task 2.1\Data\CTbrain50.nrrd")
    # save_path=r"C:\Users\Audit\Uppsala - Masters Europe\Semester 3\Project in Image analysis - Software\UCAN-PET-CT-image-data-handling-pipeline\Task 2.1\Data\Sample 2D projections\CTbrain50"
    # get_2D_projections(v_img,'CT','LT','max',15,img_n=save_path)