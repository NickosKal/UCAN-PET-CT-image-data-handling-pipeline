# UCAN-PET-CT-image-data-handling-pipeline
Project work done for the course Project in Image Analysis and Machine Learning under the Masters programme @ Uppsala University, in collaboration with the Department of Radiology.

Directory Structure: 

```bash
│───   .gitignore
│───   LICENSE
│───   README.md
│─── requirements.txt
│
├───Archive
│   ├───Task 1.1
│   │       Task1.1_convertdicom_to_volumebasedformat_nifti_UCAN.ipynb
│   │
│   ├───Task 1.2
│   │       check_for_distorted_images.ipynb
│   │       choose_the_correct_folders.py
│   │       distorted.py
│   │       Image_Selection.ipynb
│   │       Image_Selection.py
│   │       Image_Selection_ashish.ipynb
│   │       Image_Selection_ashish.py
│   │       Image_Selection_ashish_linux.py
│   │       Image_Selection_final copy.py
│   │       Image_Selection_final.py
│   │       read_imgdir.ipynb
│   │       Remove_distorted_images.py
│   │       Remove_distorted_images_linux.py
│   │       testpatientsondisk1and2.ipynb
│   │
│   ├───Task 1.3
│   │       Resample_images_from_UCAN_sample.ipynb
│   │       Resample_SUV_and_CT_images.ipynb
│   │       T2_Resample_SUV_and_CT_images.ipynb
│   │
│   ├───Task 2.1
│   │       generate_projections_for_selected_images.ipynb
│   │       sample_2D_projections.py
│   │       SimpleITK_exploration.ipynb
│   │       SimpleITK_exploration_test_to_find_distorted_images.ipynb
│   │       utils.py
│   │
│   ├───Task 3.1
│   │       Task3.1_ExtractMetaData_UCAN.ipynb
│   │       Task3.1_ExtractMetaData_UCAN_wd.ipynb
│   │
│   ├───Task 4
│   │       generate_df_multi_angled_multi_channel_projections.ipynb
│   │       T4_Generate_multi_angle_and_channel_projections.ipynb
│   │       T5_Generate_collage_from_multi_angled_projections.ipynb
│   │       T6_Generate_collage_from_multi_angled_projections.ipynb
│   │
│   └───Utils
│           config.yaml
│           utils.py
│           __init__.py
│
├───Data_processing
│       T1_Patient_examination_selection.py
│       T2_Resample_SUV_and_CT_images.py
│       T4_Generate_multi_angle_and_channel_projections.py
│       T5_Reshape_the_raw_projections.py
│       T6_Generate_collage_from_multi_angled_projections.py
│       T7_Find_arrays_with_nan_values.py
│       T8_Prepare_dataset_for_model_training.py
│       utils.py
│       __init__.py
│
├───Task 1.3
├───Task2_1
│   │   sample_2D_projections.py
│   │   SimpleITK_exploration.ipynb
│   │   utils.py
│
├───Task3_1
│       Task3.1_ExtractMetaData_UCAN.ipynb
│       Task3.1_LinkToClinicalData.ipynb
│
├───Task4
│   │   generate_final_dataset.ipynb
│   │   utils.py
│   │
│   ├───cross_validation
│   │       classification.py
│   │       classification_ashish.py
│   │       generate_dataset.py
│   │       regression.py
│   │       regression_ashish.py
│   │
│   └───Data_preparation
│           evaluate_models.ipynb
│           generate_collages.ipynb
│           reshape_collages.ipynb
│
├───testing_scripts
│       SimpleITK_exploration_test_to_find_distorted_images.ipynb
│       testing_SimpleITK_exploration.ipynb
│       testpatientsondisk1and2.ipynb
│       test_for_distorted_images.ipynb
│       test_projection_generator.ipynb
│       test_sample_2D_projections.py
│
└───Utils
        config.yaml
        utils.py
        __init__.py
```
