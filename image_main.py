from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

import tarfile
import urllib
from sklearn.model_selection import train_test_split
import shutil

from PIL import Image, ImageDraw, ImageFont
import cv2
from datetime import datetime
import random
import string

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

def cnn_step(img):
    # input data
    cv2.imwrite('inputdata/data/example.jpg', img)

    # make dataloader
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    root_dir = './inputdata'

    data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    dataset = datasets.ImageFolder(root=root_dir, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    class_label = [ "abyssinian",
                    "american_bulldog",
                    "american_pit_bull_terrier",
                    "basset_hound",
                    "beagle",
                    "bengal",
                    "birman",
                    "bombay",
                    "boxer",
                    "british_shorthair",
                    "chihuahua",
                    "egyptian_mau",
                    "english_cocker_spaniel",
                    "english_setter",
                    "german_shorthaired",
                    "great_pyrenees",
                    "havanese",
                    "japanese_chin",
                    "keeshond",
                    "leonberger",
                    "maine_coon",
                    "miniature_pinscher",
                    "newfoundland",
                    "persian",
                    "pomeranian",
                    "pug",
                    "ragdoll",
                    "russian_blue",
                    "saint_bernard",
                    "samoyed",
                    "scottish_terrier",
                    "shiba_inu",
                    "siamese",
                    "sphynx",
                    "staffordshire_bull_terrier",
                    "wheaten_terrier",
                    "yorkshire_terrier" ]

    # set model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_label))
    model = model.to(device)

    model.load_state_dict(torch.load('kadai-weight-cpu.pth'))
    print("read weight")

    # input data
    model.eval()
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        print("preds :", class_label[preds])
        print("\n")

    # write name
    writetext = class_label[preds]
    img = Image.open('inputdata/data/example.jpg')
    imgsize = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Arial Unicode.ttf', 75);
    size = font.getsize(writetext)

    pos = (imgsize[0] - size[0], imgsize[1] - size[1])
    bw = 2.1
    # 文字の縁取り
    draw.text((imgsize[0]-size[0]-bw, imgsize[1]-size[1]-bw), writetext, font=font, fill=(255, 0, 0))
    draw.text((imgsize[0]-size[0]-bw, imgsize[1]-size[1]+bw), writetext, font=font, fill=(255, 0, 0))
    draw.text((imgsize[0]-size[0]+bw, imgsize[1]-size[1]-bw), writetext, font=font, fill=(255, 0, 0))
    draw.text((imgsize[0]-size[0]+bw, imgsize[1]-size[1]+bw), writetext, font=font, fill=(255, 0, 0))
    # 文字
    draw.text(pos, writetext, font=font, fill='#FFF')
    print("save_img")

    #SAVE_DIR = "./images/"
    SAVE_DIR = "./static/images/"
    #dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") + random_str(5)
    #save_path = os.path.join(SAVE_DIR, dt_now + ".jpg")
    save_path = os.path.join(SAVE_DIR, "kekka.jpg")

    img.save(save_path, 'JPEG', quality=100, optimize=True)
