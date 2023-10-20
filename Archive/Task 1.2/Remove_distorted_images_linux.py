
import os
import sys
import time
from datetime import datetime
import psutil
import numpy  as np
import pandas as pd
import concurrent.futures
import pydicom as dicom
import SimpleITK as sitk
import random
random.seed(10)

dicom.config.convert_wrong_length_to_UN = True

def checkDistortedImg(vol_img,ptype='mean',angle=90):
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

    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]

    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle) 
        resampled_image = sitk.Resample(image1=vol_img,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkLinear,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection = [1,0,0,0,1,0,0,0,1],
                                        defaultPixelValue =  -20, #HU unit for air in CT, possibly set to 0 in other cases
                                        outputPixelType = vol_img.GetPixelID())
        proj_image = projection[ptype](resampled_image, paxis)
        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0
        sitk.Extract(proj_image, extract_size)

def outputDistortedImg(df):
    pid  = os.getpid()
    ppid = os.getppid()
    start = time.time()
    print("PPID %s->%s Started on %s"%(ppid, pid, str(datetime.now())))
    
    exception_lst = []

    for _, row in df.iterrows():
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(row['directory'])
        reader.SetFileNames(dicom_names)
        vol_img = reader.Execute()
        try:
            checkDistortedImg(vol_img)
        except:
            exception_lst.append(row['directory'])

    end = time.time()

    print("PPID %s Completed in %s"%(os.getpid(), round((end-start)/60,2)))

    return exception_lst

if __name__ == '__main__':

    #source_path = r"F:\U-CAN-Lymfom_A"
    source_path = r"/media/andres/T7 Shield/ucan_lymfom"
    # source_path = "Selected_for_Sorting_test/"
    #destination_path = r"F:\U-CAN-Lymfom_A\Selected_for_UCAN_project\DistortedImageTest\ParallelRun"
    destination_path = r"/media/andres/T7 Shield/ucan_lymfom/DistortedImageTest/ParallelRun"

    try:
        os.makedirs(destination_path, exist_ok=True)
    except OSError as error:
        print(error)

    print("Reading the directory tree")
    directory_list = list()
    for root, dirs, files in os.walk(source_path, topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))

    print("Total directories in source list: ", len(directory_list))

    remove_lst = ['PR----BONE-PULM-mm',  'PR----Lunga-0.6-ax-mm',  'PR----WB-Venfas-0.6-ax-mm',  'PR----LUNG-1.25-AX-mm', 
                   'PR----WB-Ben-lunga-0.6-ax-mm',  'PR----WB-Venfas-3-ax-mm',  'PR----LUNG-1.25-AX-mm',  'PR----BONE-1.25-AX-mm', 
                   'PR----LUNG-1.25-AX-mm',  'PR----Lunga-0.6-ax-mm',  'PR----SAVED-IMAGES-PR-mm',  'PR----e1-QCFX-S-400-Static-mm',
                   'PR----WB-Venfas-0.6-ax-mm',  'PR----WB-VEN-AX-mm',  'PR----WB-Ben-lunga-0.6-ax-mm',  'PR----LUNG-1.25-AX-mm',
                   'PR----THORAX-AX-mm', 'PR----LUNG-1.25-AX-mm', 'PR----THORAX-INANDAD-mm', 'PR----KEY_IMAGES-PR-mm', 'PR----SAVED-PR-mm', 
                   'Examinations that miss either CT or PET or both', 'MR-', 'sag', 'cor', 'ot-']

    print("Removing unnecessary directory from directory list")
    findir_lst = []
    rejection_lst = []
    for dir in directory_list:
        if len(dir.split('/'))>6 and 'ASPTCT' in dir and  all(item.lower() not in dir.lower() for item in remove_lst):
            print(dir)
            findir_lst.append(dir)
        else:
            rejection_lst.append(dir)

    print("Writing first rejected directory list")
    with open(destination_path + r'/FirstRejectionDirs.txt', 'w') as fp:
        for item in rejection_lst:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

    print("Total directories in final list: ", len(findir_lst))

    print("Loading directories into Dataframe")
    data = pd.DataFrame(findir_lst, columns=['directory'])

    logical    = False
    distorted_lst = []
    num_procs  = psutil.cpu_count(logical=logical)
    if len(sys.argv) > 1:
        num_procs = int(sys.argv[1])
    print("Number of processes available: ", num_procs)

    #print("Splitting dataframe into ", num_procs-2, " dataframes")
    #splitted_df = np.array_split(data, num_procs - 2)

    num_threads = (num_procs - 1) * 2
    print("Splitting dataframe into ", num_threads, " dataframes")
    splitted_df = np.array_split(data, num_threads)

    start = time.time()

    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
        results = [executor.submit(outputDistortedImg,df=df) for df in splitted_df]
        for result in concurrent.futures.as_completed(results):
            try:
                distorted_lst.append(result.result())
            except Exception as ex:
                print(str(ex))
                pass
    """

    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Start the load operations and mark each future with its URL
        results = [executor.submit(outputDistortedImg, df) for df in splitted_df]
        for future in concurrent.futures.as_completed(results):
            try:
                distorted_lst.append(future.result())
            except Exception as ex:
                print(str(ex))
                pass

    end = time.time()

    print("Writing final distorted images lst")
    with open(r'F:\U-CAN-Lymfom_A\Selected_for_UCAN_project\DistortedImageTest\distorted_lst.txt', 'w') as fp:
        for item in distorted_lst:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

    print("-------------------------------------------")
    print("PPID %s Completed in %s"%(os.getpid(), round(end-start,2)))