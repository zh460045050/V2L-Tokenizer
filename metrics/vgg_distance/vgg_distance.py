import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = 0
import numpy as np 
import torch 
import argparse
import pathlib
from PIL import Image
from loss_vgg2 import VGG19, vgg_loss

# You can rewrite the function of read_data 
# according to the name format of your own images.
# But the predicted image must correspond to the groundtruth images one-to-one.
# Here, the name of predicted images is the same as the groundtruth images
def read_data(pred_path, gt_path):
    pred = []
    gt = []
    path = pathlib.Path(pred_path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    # num = len(files)

    for name in files:
        pred.append(np.asarray(Image.open(os.path.join(pred_path, name))))
        gt.append(np.asarray(Image.open(os.path.join(gt_path, name))))

    pred = np.asarray(pred)
    gt = np.asarray(gt)

    # if images are gray
    if len(pred.shape) == 3:
        pred = pred[:,:,:,np.newaxis]
        pred = np.concatenate([pred,pred,pred], axis=3)

    if len(gt.shape) == 3:
        gt = gt[:,:,:,np.newaxis]
        gt = np.concatenate([gt,gt,gt], axis=3)

    pred = pred.transpose((0, 3, 1, 2))
    gt = gt.transpose((0, 3, 1, 2))

    pred = pred.astype(np.float32) / 255.0
    gt = pred.astype(np.float32) / 255.0

    return pred, gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', type=str, default=None, help='the path of predicted results')
    parser.add_argument('-t', '--target', type=str, default=None, help='the path of corresponding groundtruth')
    parser.add_argument('-m', '--mode', type=int, default=2)
    args = parser.parse_args()

    pred_path = args.pred
    gt_path = args.target
    print('Process: ' + pred_path)

    print('read data...')
    pred, gt = read_data(pred_path, gt_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # build VGG model
    print('VGG build...', flush=True)
    model_vgg = VGG19().to(device)
    
    distance = []
    for i in range(gt.shape[0]):
        pred_img = pred[i]
        gt_img = gt[i]
        pred_img = torch.tensor(pred_img).to(device)
        gt_img = torch.tensor(gt_img).to(device)
        out1 = model_vgg(pred_img)
        out2 = model_vgg(gt_img)
        loss_vgg = vgg_loss(out1, out2, args.mode)
        distance.append(loss_vgg)
        print('image %d, vgg_distance = %.6f' % (i, loss_vgg))
    ave = sum(distance) / len(distance)
    print('average vgg_distance = %.6f' % (ave))
    print('Done')