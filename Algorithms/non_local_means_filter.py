import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

image = cv.imread('../Images/noisyImage_Gaussian_01.jpg', 0)


def non_local_filter(noisy, h, templateWindowSize, searchWindowSize, verbose=True):
    padding_size = searchWindowSize // 2
    image_local = noisy.copy()
    height, width = noisy.shape

    padded_image = np.pad(noisy, pad_width=padding_size, mode='reflect')

    outputImage = padded_image.copy()

    smallhalfwidth = templateWindowSize // 2

    for imageX in range(padding_size, padding_size + width):
        for imageY in range(padding_size, padding_size + height):

            bWinX = imageX - padding_size
            bWinY = imageY - padding_size

            local_patch = padded_image[imageX - smallhalfwidth:imageX + smallhalfwidth + 1,
                          imageY - smallhalfwidth:imageY + smallhalfwidth + 1]

            pixelColor = 0
            totalWeight = 0

            for sWinX in range(bWinX, bWinX + searchWindowSize - padding_size):
                for sWinY in range(bWinY, bWinY + searchWindowSize - padding_size):
                    template_patch = padded_image[sWinX:sWinX + templateWindowSize, sWinY:sWinY + templateWindowSize]
                    euclideanDistance = np.sqrt(np.sum(np.square(template_patch - local_patch)))
                    weight = np.exp(- euclideanDistance / h)
                    totalWeight += weight
                    pixelColor += weight * padded_image[sWinX + smallhalfwidth, sWinY + smallhalfwidth]

            pixelColor /= totalWeight
            if pixelColor > 255:
                pixelColor = 255
            outputImage[imageX, imageY] = pixelColor

    return outputImage[padding_size:padding_size + width,
           padding_size:padding_size + height]


def gaussianBlurOpencv(image, boxSize, sigma):
    gaussian_blur = cv.GaussianBlur(image, (boxSize, boxSize), sigma)
    gaussian_blur = cv.normalize(gaussian_blur, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                 dtype=cv.CV_32F).astype(np.uint8)
    return gaussian_blur


if __name__ == '__main__':
    filtered_image_OpenCV = cv.fastNlMeansDenoising(image, h=12, templateWindowSize=5, searchWindowSize=23)
    print("OpenCV Filtered Image Completed")
    print(filtered_image_OpenCV)
    filtered_image = non_local_filter(image, 16, 7, 21, True)
    filtered_image = filtered_image.astype(np.uint8)
    print("My Non Local Filtered Image Completed")

    gaussian_blur5 = gaussianBlurOpencv(image, 5, 0)

    cv.imshow('Original Image', image)
    cv.imshow('My NLM Filter', filtered_image)
    cv.imshow('Opencv Gaussian', gaussian_blur5)
    cv.imshow('Opencv NLM Filter', filtered_image_OpenCV)
    cv.waitKey(0)
    cv.destroyAllWindows()
