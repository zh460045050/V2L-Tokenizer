import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import os 
import pathlib
import argparse
from PIL import Image
import cv2

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index]

    def __len__(self):
        return len(self.orig)

def read_data(pred_path):
    pred = []
    path = pathlib.Path(pred_path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.JPEG'))
    # num = len(files)

    for name in files:
        img = cv2.imread(str(name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pred.append(np.asarray(img))
        #try:
        #    import cv2
        #    cv2.imread(name)
            #Image.open(name)
        #except:
        #    print("error")
        #pred.append(np.asarray(Image.open(name).convert("RGB")))

    pred = np.asarray(pred)
    # if images are gray
    if len(pred.shape) == 3:
        pred = pred[:,:,:,np.newaxis]
        pred = np.concatenate([pred,pred,pred], axis=3)
    pred = pred.transpose((0, 3, 1, 2))
    pred = pred.astype(np.float32) / 255.0

    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=None)
    args = parser.parse_args()
    
    print('Read data...')
    data = read_data(args.path)

    ite_data = IgnoreLabelDataset(data)

    print ("Calculating Inception Score...")
    print (inception_score(ite_data, cuda=True, batch_size=10, resize=True, splits=10))
