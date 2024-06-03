import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def model_init():
    categories = [
        "Asparagus",
        "Carrot",
        "Oysters",
        "Pork",
        "Salmon",
        "Zuccini",
        "Strawberries",
        "Sausages",
        "Garlic",
        "Ginger",
        "Cauliflower",
        "Capsicum",
        "Pumpkin",
        "Rockmelon",
        "Watermelon",
        "Avocado",
        "Tomato",
        "Pineapple",
        "Pears",
        "Apples",
        "Peach",
        "Trout",
        "Snapper",
        "Barramundi",
        "Prawns",
        "Tropical Fish",
        "Steak",
        "Chicken",
        "Lamb",
        "Mushrooms",
        "Red Onion",
        "Tortellini",
        "Blueberries",
        "Lettuce",
        "Milk",
        "Eggs",
        "Juice",
        "Kiwi",
        "Butter",
        "Cheese"
    ]

    image_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    modelRes50 = torch.load('./savedModels/resnet50_A.pth')
    modelRes18 = None
    modelSqueeze = None
    modelVGG = None
    

    return modelRes50, modelRes18, modelSqueeze, modelVGG, image_transforms, categories

def classify(model, image_transforms, image_path, categories):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print(categories[predicted.item()])