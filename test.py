import torch
import torch.nn.functional as F
import numpy as np
import os
import imageio
import argparse
from scipy import misc
from Code.data import test_dataset
from Code.BSNet_Res2Net import BSNet as Network



def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--data_path', type=str, default='./TestingSet/',
                        help='Path to test data')

    parser.add_argument('--pth_path', type=str, default='./weights/BSNet.pth',
                        help='Path to weights file.')

    parser.add_argument('--save_path', type=str, default='./result/',
                        help='Path to save the predictions.')
    opt = parser.parse_args()

    model = Network()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    image_root = '{}/Imgs/'.format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)
    os.makedirs(opt.save_path, exist_ok=True)

    for i in range(test_loader.size):
        image, name = test_loader.load_data()

        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

        res = lateral_map_2
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(opt.save_path + name, res)

    print('Test Done!')


if __name__ == "__main__":
    inference()
