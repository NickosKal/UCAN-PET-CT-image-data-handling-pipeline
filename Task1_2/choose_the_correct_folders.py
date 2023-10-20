import glob
import os

path = "Selected_for_Sorting_test/"
# for folder_path in glob.glob(path):
#     print(folder_path)
# for folder_path, folders, files in os.walk(path):
#     print(folders)
ct_found = False
pt_found = False

directory_list = list()
for root, dirs, files in os.walk(path, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))

for folder_path in directory_list:
    first_split_of_path = folder_path.split("/")
    second_part_of_path = first_split_of_path[1]
    # print(second_part_of_path)
    second_split_of_path = second_part_of_path.split("\\")
    exams = os.path.join(path, second_split_of_path[0])
    # print(exams)
    # print(os.listdir(exams))
    examinations = os.listdir(os.path.join(path, second_split_of_path[0]))
    # print(examinations)
    for item in examinations:
        print(item)


