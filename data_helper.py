import glob
import os

# folders = glob.glob("../embodied-learning-data-test/train/12/", recursive=True)
folders = glob.glob("../embodied-learning-data-test/**/", recursive=True)

for folder in folders:
    files = glob.glob(folder + "**.jpg")

    for file in files:
        # # Extract the last folder name
        last_folder = os.path.basename(os.path.dirname(folder))

        # Extract the filename
        filename = os.path.basename(file)
        new_file_name = folder + last_folder + "_" + filename
        print(new_file_name)
        os.rename(file, new_file_name)

        print(file)
