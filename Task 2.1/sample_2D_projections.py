import SimpleITK as sitk
import matplotlib.pyplot as plt





if __name__=="__main__":
    IMG_PATH=r"C:\Users\Audit\Uppsala - Masters Europe\Semester 3\Project in Image analysis - Software\UCAN-PET-CT-image-data-handling-pipeline\Task 2.1\Data\sample3Dimg.nrrd"
    vol_img = sitk.ReadImage()
    file_reader = sitk.ImageFileReader(IMG_PATH)
    file_reader.SetImageIO('NrrdImageIO')
    file_reader.SetFileName(IMG_PATH)

# plt.imshow(sitk.GetArrayViewFromImage(logo))
# plt.axis('off')
