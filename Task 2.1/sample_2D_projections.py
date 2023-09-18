import SimpleITK as sitk
import matplotlib.pyplot as plt

vol_img = sitk.Image()

file_reader = sitk.ImageFileReader()
file_reader.SetImageIO('NrrdImageIO')
file_reader.SetFileName('Task 2.1\Data\sample3Dimg.nrrd')
file_reader.Execute()

# plt.imshow(sitk.GetArrayViewFromImage(logo))
# plt.axis('off')
