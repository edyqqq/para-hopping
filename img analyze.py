from PIL import Image
import numpy as np
import skimage.measure
import math
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim


def sumofpixel(height,width,img1,img2):
    matrix=np.empty([width, height])
    for y in range(0,height):
        for x in range(0,width):
            # print(img1[x,y],img2[x,y])
            if img1[x,y] == img2[x,y]:
                matrix[x,y]=0
            else:
                matrix[x,y]=1
    psum=0
    for y in range(0,height):
        for x in range(0,width):
            psum=matrix[x,y]+psum
    return psum

def npcr(img1,img2):
    
    height = img1.shape[0]
    width = img1.shape[1]

    npcrv = ((sumofpixel(height,width,img1,img2)/(height*width))*100)
    return npcrv

def uaci(img1,img2):
    height,width = img1.shape
    value = 0
    for y in range(height):
        for x in range(width):
            value += (abs(int(img1[x,y])-int(img2[x,y])))

    value = value*100/(width*height*255)
    return value       


def main():
    img1 = cv2.imread('./Lenna_256.bmp',0)
    img2 = cv2.imread('./test result/hopped_256/lena_en_c2d1p16.png',0)
    img3 = cv2.imread('./test result/hopped_256/lena_en_c2d1p16k2.png',0)
    print(np.sum(img2-img3))
    PSNR = peak_signal_noise_ratio(img1, img2)
    SSIM = ssim(img1, img2, multichannel=True)
    print(f'PSNR: {PSNR}')
    print('NPCR:',npcr(img2,img3))
    print('Entropy:',skimage.measure.shannon_entropy(img2))
    print('SSIM:', SSIM)
    print('UACI:',uaci(img2, img3))

if __name__ == '__main__':
    main()        
        