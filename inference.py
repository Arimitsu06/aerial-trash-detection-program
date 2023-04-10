import mmcv
import mmdet
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import os
import glob
import argparse

parser = argparse.ArgumentParser(
                    prog='Inference',
                    description='Inferences on a set of images in a folder and saves the results in image format in another folder')

parser.add_argument('-m', help='path to model checkpoint file (Default: ./model_checkpoints/epoch_4.pth)', default='./model_checkpoints/epoch_4.pth', type=str)
parser.add_argument('-c', help='path to model config file (Default: ./config.py)', default='./config.py', type=str)
parser.add_argument('-d', help='path to folder to save inference in (Default: ./results)', default='./results', type=str)
parser.add_argument('-i', help='path to folder of images to inference on (Default: ./test_images)', default='./test_images', type=str)
parser.add_argument('-f', help='image file format(.jpg, .png, etc) (Default: .jpg)', default='.jpg', type=str)
args = parser.parse_args()

def main():
    cfg = Config.fromfile(args.c)
    model = init_detector(cfg, args.m, device='cpu')

    for file in glob.glob(f'{args.i}/*{args.f}'):
        print('Inferencing on ' + file)
        img = mmcv.imread(file)
        result = inference_detector(model, img)
        model.show_result(img, result, out_file=f"{args.d}/{file.split('/')[-1]}")

if __name__ == "__main__":
    main()
