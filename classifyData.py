#We receive data in the format classID_className_individualID

#the classId is a number from 0 to 39, 
#className is a food type like asparagus, carrots, pork etc, 
#individualID is a 3 digit number detailing which image of a given class we're working with 
    #eg aparagus_005 and pears_005 are the 5th aparagus and pear image respectively, 

#we do not necessarily need 1..n of each image and there are break for example we have tomato 001 002 007 and 009

import numpy as np
import os

print("Export images into their own folder with no other files in the datasets folder and enter folder name.\nIf name is left empty will default to FoodTest1.")
folderName = input("Enter Folder Name:")

if not folderName:
    folderName = "FoodTest1"

directory = ".\\datasets\\"+folderName

#we error if we have files in the folder that do not fit the format but I'm gonna specifically catch .DS_Store because the initial dataset includes it
i = 0
with os.scandir(directory) as root_dir:
    for path in root_dir:
        if path.is_file():
            if (path.name != ".DS_Store"):
                i += 1
                #print(f"Full path is: {path} and just the name is: {path.name}")

print(f"{i} files scanned successfully.\n")
imageList = np.empty([i,4], dtype=object)

i = 0
with os.scandir(directory) as root_dir:
    for path in root_dir:
        if path.is_file():
            if (path.name != ".DS_Store"):
                values = path.name.split('_')
                imageList[i] = [values[0], values[1], values[2], path.path ]
                imageList[i,2] = imageList[i,2].removesuffix('.JPG')
                i += 1

print(imageList)
np.savetxt("result.csv", imageList, delimiter=",", header="", comments="",fmt='%s')
