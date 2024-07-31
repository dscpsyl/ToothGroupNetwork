import torch
import numpy as np

import argparse as ap
import os.path as path

import cv2 as c
from tqdm import tqdm

import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision

import losses as l

def train(model, dataset, outpath):
    
    NEPOCHS = 100
    LR = 0.001
    BATCHSIZE = 1
    
    lossFN = l.GeneralizedDiceLoss()
    # lossFN = torchvision.ops.sigmoid_focal_loss
    # lossFN = torch.nn.functional.binary_cross_entropy
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    data = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)
    
    for _ in tqdm(range(NEPOCHS)):
        
        for image, label in data:
            optimizer.zero_grad()
        
            pred = model(image[0])
            pred = _filter(pred)
            loss = lossFN(pred, label[:, :, :, :, 0])
            
            # layer = pred[0, 0, :, :].cpu().detach().numpy()
            # p = np.stack((layer, layer, layer), axis=2)
            
            # layer = label[0, 0, :, :, 0].cpu().detach().numpy()
            # lab = np.stack((layer, layer, layer), axis=2)
            
            # c.imshow(f"Pred ", p * 255)
            
            # c.imshow(f"Label ", lab * 255)
            
            # c.waitKey(0)
            # c.destroyAllWindows()
            
            # print(loss)
            
            loss.backward()
            optimizer.step()
    
    
    # Save as a torch script
    tsModel = torch.jit.script(model)
    tsModel.save(outpath + ".pth")
    

def test(model, data, outPath):
    
    with torch.no_grad():
        pred = model(data)

    # t = pred.cpu().detach().numpy()
    # print(t.shape)
    # np.savetxt("pred.txt", t[0, 0, :, :])
        

    pred = _filter(pred)
    
    torch.save(pred, outPath)
    
    pred = pred[0, 0, :, :].cpu().detach().numpy()
    # assert np.any(pred == 1), f"test::No are of the mask is 1. Did the model predict anything?"
    pred = pred * 255
    img = np.stack((pred, pred, pred), axis=2)
    
    c.imshow("Pred Mat", img)
    c.waitKey(0)
    c.destroyAllWindows()


def _filter(data):
    return torch.round(data)        

class MRIDataset(Dataset):
    
    def __init__(self, dataDir, dataNamesFile, transform=None, target_transform=None):
        
        assert path.isdir(dataDir), f"MRIDataset::{dataDir} is not a directory or does not exist"
        self.dataPath = dataDir
        
        assert path.isfile(dataNamesFile), f"MRIDataset::{dataNamesFile} is not a file or does not exist"
        with open(dataNamesFile, 'r') as f:
            self.dataNames = f.readlines()
        
        for name in self.dataNames:
            assert path.isfile(path.join(dataDir, name.strip()) + ".pt"), f"MRIDataset::{name.strip()}'s .pt file does not exist. Did you preprocess the data?"
            assert path.isfile(path.join(dataDir, name.strip()) + "-gt.pt"), f"MRIDataset::{name.strip()}'s ground truth .pt file does not exist. Did you preprocess the data?"
        
        
        self.transform = transform
        self.target_transform = target_transform
        
        print(self)
        
    def __len__(self):
        return len(self.dataNames)
        
    def __getitem__(self, idx):
        p = path.join(self.dataPath, self.dataNames[idx].strip())
        image = torch.load(p + ".pt")
        gt = torch.load(p + "-gt.pt")
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt = self.target_transform(gt)
        
        return image, gt
        
    def __repr__(self):
        return f"MRIDataset({self.dataPath}) of length {len(self)}"
    

def main(args):
    
    if args.model is not None:
        model = torch.jit.load(args.model)
    else:
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                in_channels=3, out_channels=1, init_features=32, pretrained=True)
        
    outPath = args.output
    
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    if args.train:
        model.train()
        data = MRIDataset(args.data, args.ground_truth)
        
        train(model, data, outPath)
    else:
        model.eval()
        tData = torch.load(args.data)
        
        test(model, tData, outPath)
    
    
        
if __name__ == "__main__":
    parser = ap.ArgumentParser(description="MRI Segmentation Model")
    parser.add_argument("-m", "--model", help="The path to the model", type=str)
    parser.add_argument("--train", help="Train the model", action="store_true")
    parser.add_argument("-d", "--data", help="The path to the tensor data", required=True, type=str)
    parser.add_argument("-g", "--ground-truth", help="The path to the ground truth data", type=str)
    parser.add_argument("-o", "--output", help="The path to the output", required=True, type=str)
    
    args = parser.parse_args()
    
    main(args)
    