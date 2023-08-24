from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import numpy as np
import os
from glob import glob



def read_mimic(batchsize,data_dir = '../mimic_part_jpg'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-5,5)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {x: dataset(mode=x, transform=data_transforms[x])
                      for x in ['train', 'test']}
    
    data_loader_train = DataLoader(dataset=image_datasets['train'],
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True
                                   )
    data_loader_test = DataLoader(dataset=image_datasets['test'],
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )
    
    return data_loader_train,data_loader_test

class dataset(Dataset):

    def __init__(self, data_dir='../mimic_part_jpg', mode="train", transform=None):

        self.root = data_dir
        self.mode = mode
        self.T = transform
        self.csv = pd.read_csv(os.path.join(self.root, "gaze", "fixations.csv"))
        self.labels = ["CHF", "Normal", "pneumonia"]
        self.labelsdict = {"CHF": 0, "Normal": 1, "pneumonia": 2}
        self.idlist = []
        for i in range(len(self.labels)):
            self.idlist.extend(glob(os.path.join(self.root, self.mode, self.labels[i], "*.jpg")))
        
    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, idx):

        # get path
        imgpath = self.idlist[idx]
        id = imgpath.split("/")[-1].split(".jpg")[0]
        gazepath = os.path.join(self.root, "gaze", "fixations", "{}.npy".format(id))
        

        # extract image
        with open(imgpath, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")


        # extract label
        label = self.labelsdict[imgpath.split("/")[-2]]

        # extract gaze
        id = imgpath.split("\\")[-1].split(".jpg")[0]
        gaze = np.zeros((img.size[1], img.size[0]), dtype=np.float32)
        idcsv = self.csv.loc[self.csv["DICOM_ID"] == id]
        for i in range(len(idcsv)):
            if i == 0:
                t = idcsv.iloc[i]["Time (in secs)"]
            else:
                t = idcsv.iloc[i]["Time (in secs)"] - idcsv.iloc[i-1]["Time (in secs)"]
            x = idcsv.iloc[i]["X_ORIGINAL"]
            y = idcsv.iloc[i]["Y_ORIGINAL"]
            gaze[y,x] = t
        gaze = np.log(gaze+0.01)
        gaze = (((gaze-gaze.min())/(gaze.max()-gaze.min())) * 255).astype(np.uint8)
        gimg = gaze[..., np.newaxis].repeat(3, axis=2)
        gimg = Image.fromarray(gimg)


        # transform
        state = torch.get_rng_state()
        img = self.T(img)
        img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        torch.set_rng_state(state)
        gaze = self.T(gimg)
        gaze = self.getPatchGaze(gaze[0])


        return img, label, gaze

    def getPatchGaze(self, gaze):
        g = np.zeros((56,56), dtype=np.float32)
        for i in range(56):
            for j in range(56):
                x1 = 4*i-7
                x2 = 4*i+7
                y1 = 4*j-7
                y2 = 4*j+7
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > 223:
                    x2 = 223
                if y2 > 223:
                    y2 = 223
                g[i,j] = gaze[x1:x2, y1:y2].sum()
        if g.max()-g.min() != 0:
            g = (g-g.min())/(g.max()-g.min())
        return g


