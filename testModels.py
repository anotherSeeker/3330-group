#We receive data in the format classID_className_individualID

#the classId is a number from 0 to 39, 
#className is a food type like asparagus, carrots, pork etc, 
#individualID is a 3 digit number detailing which image of a given class we're working with 
    #eg aparagus_005 and pears_005 are the 5th aparagus and pear image respectively, 

#we do not necessarily need 1..n of each image and there are break for example we have tomato 001 002 007 and 009

import numpy as np
import os
import shutil
from loadModel import model_init, classify

print("Export images into ***their own folder*** with no other files ***in the datasets folder*** and enter folder name with no /'s.\nIf name is left empty will default to defaultTestImages.")
folderName = input("Enter Folder Name:")

if not folderName:
    folderName = "defaultTestImages"

directory = "./datasets/"+folderName

#frankly, I have no idea if this will matter but we're catching .ds_store
i = 0
with os.scandir(directory) as root_dir:
    for path in root_dir:
        if path.is_file():
            if (path.name != ".DS_Store"):
                i += 1
                #print(f"Full path is: {path} and just the name is: {path.name}")

print(f"{i} files scanned successfully.\n")

testDirectory = './datasets/test images'

if not os.path.exists(testDirectory):
    os.makedirs(testDirectory)
    print("made "+testDirectory)

#wipe the test directory
for filename in os.listdir(testDirectory):
    file_path = os.path.join(testDirectory, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

#copy images to test directory
i = 0
with os.scandir(directory) as root_dir:
    for path in root_dir:
        if path.is_file():
            if (path.name != ".DS_Store"):
                shutil.copy2(path, testDirectory)
                i += 1

print("Images moved to ./datasets/test images")

looping = True
res50,res18,squeeze,vgg,image_transforms, categories = model_init()

while (looping == True):
    desiredModel = input("Enter 0, 1, 2 to test saved models\nResnet50, Resnet18, or Squeezenet: ")
    model = None

    match desiredModel:
        case '0':
            print("ResNet50")
            model = res50
        case '1':
            print("ResNet18")
            model = res18
        case '2':
            print("Squeezenet")
            model = squeeze
        case _:
            looping = False
        
    if (looping != False):
        with os.scandir(testDirectory) as root_dir:
            for path in root_dir:
                if path.is_file():
                    #print(path.path)
                    classify(model, image_transforms, path.path, categories)
        
        print("Enter anything else to close the program")
else:
    print("Goodbye")

        
