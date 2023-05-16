from san_api import SanLandmarkDetector
import torch
import cv2 as cv

if __name__ == '__main__':
    model_path = 'D:/Users/Desktop/workspace/preproccess/checkpoint/san_checkpoint_49.pth.tar'
    image_path = './data/S011/000.jpg'
    face = (143.,110.,353.,383.)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    det = SanLandmarkDetector(model_path, device)
    # locs, scores = det.detect(image_path, face)
    # a = 1
    # img = cv.imread(image_path)
    # h, w, _= img.shape
    # cv.rectangle(img, (143,110), (353,383), (0,255,0), 4)
    # for loc in locs:        
    #     cv.circle(img, (loc[0], loc[1]), 2, (0, 0, 255), 0)
    # cv.imshow('test', img)
    # cv.waitKey(0)
