import numpy as np
import cv2 as cv

noisyImage_gaussian = cv.imread('../Images/noisyImage_Gaussian.jpg', 0)
lower_gaussian_noise_image = cv.imread('../Images/noisyImage_Gaussian_01.jpg', 0)
golden_image = cv.imread('../Images/lena_grayscale_hq.jpg', 0)

noisyImage_gaussian = cv.normalize(noisyImage_gaussian, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,
                                   dtype=cv.CV_32F)
lower_gaussian_noise_image = cv.normalize(lower_gaussian_noise_image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,
                                          dtype=cv.CV_32F)


def adaptiveMeanFilter(noise_variance):
    gaussian_array = noisyImage_gaussian
    denoised_array = np.zeros((len(gaussian_array), len(gaussian_array)))
    array_with_padding = np.pad(gaussian_array, pad_width=2, mode='edge')
    for i in range(2, len(array_with_padding) - 2):
        for j in range(2, len(array_with_padding) - 2):
            Sxy = array_with_padding[i - 2:i + 3, j - 2:j + 3]
            denoised_array[i - 2][j - 2] = (array_with_padding[i][j] -
                                            ((noise_variance / np.var(Sxy))
                                             * (array_with_padding[i][j] - np.average(Sxy))))

    denoised_array = cv.normalize(denoised_array, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                  dtype=cv.CV_32F).astype(np.uint8)
    return denoised_array


def boxFilterOpencv(image, boxSize):
    box_filtered_array = cv.blur(image, (boxSize, boxSize), borderType=cv.BORDER_REPLICATE)
    box_filtered_array = cv.normalize(box_filtered_array, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                      dtype=cv.CV_32F).astype(np.uint8)
    return box_filtered_array


def bilateralOpencv(image, diameter, sigmaColor, sigmaSpace):
    bilateral_cv = cv.bilateralFilter(image, diameter, sigmaColor, sigmaSpace, borderType=cv.BORDER_REPLICATE)
    bilateral_cv = cv.normalize(bilateral_cv, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                dtype=cv.CV_32F).astype(np.uint8)
    return bilateral_cv


def gaussianBlurOpencv(image, boxSize, sigma):
    gaussian_blur = cv.GaussianBlur(image, (boxSize, boxSize), sigma)
    gaussian_blur = cv.normalize(gaussian_blur, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                 dtype=cv.CV_32F).astype(np.uint8)
    return gaussian_blur


def q1(noisy_image):
    box_filtered_array3 = boxFilterOpencv(noisy_image, 3)
    box_filtered_array5 = boxFilterOpencv(noisy_image, 5)
    gaussian_blur3 = gaussianBlurOpencv(noisy_image, 3, 0)
    gaussian_blur5 = gaussianBlurOpencv(noisy_image, 5, 0)
    adaptive_mean = adaptiveMeanFilter(0.0042)
    bilateral_cv = bilateralOpencv(noisy_image, 5, 3, 0.9)
    my_bilateral = bilateral_filter(noisy_image, 5, 3, 0.9)


    cv.imshow('3x3 box filter PSNR:{}'.format(cv.PSNR(golden_image, box_filtered_array3)), box_filtered_array3)
    cv.imshow('5x5 box filter PSNR:{}'.format(cv.PSNR(golden_image, box_filtered_array5)), box_filtered_array5)
    cv.imshow('3x3 gaussian filter PSNR:{}'.format(cv.PSNR(golden_image, gaussian_blur3)), gaussian_blur3)
    cv.imshow('5x5 gaussian filter PSNR:{}'.format(cv.PSNR(golden_image, gaussian_blur5)), gaussian_blur5)
    cv.imshow('adaptive mean filter PSNR:{}'.format(cv.PSNR(golden_image, adaptive_mean)), adaptive_mean)
    cv.imshow('Opencv bilateral filter PSNR:{}'.format(cv.PSNR(golden_image, bilateral_cv)), bilateral_cv)
    cv.imshow('My bilateral filter PSNR:{}'.format(cv.PSNR(golden_image, my_bilateral)), my_bilateral)
    cv.waitKey(0)
    cv.destroyAllWindows()


def q2(noisy_image):
    box_filtered_array3 = boxFilterOpencv(noisy_image, 3)
    box_filtered_array5 = boxFilterOpencv(noisy_image, 5)
    gaussian_blur3 = gaussianBlurOpencv(noisy_image, 3, 0)
    gaussian_blur5 = gaussianBlurOpencv(noisy_image, 5, 0)
    adaptive_mean = adaptiveMeanFilter(0.0042)
    bilateral_cv = bilateralOpencv(noisy_image, 3, 0.1, 1)
    my_bilateral = bilateral_filter(noisy_image, 3, 0.1, 1)


    cv.imshow('3x3 box filter PSNR:{}'.format(cv.PSNR(golden_image, box_filtered_array3)), box_filtered_array3)
    cv.imshow('5x5 box filter PSNR:{}'.format(cv.PSNR(golden_image, box_filtered_array5)), box_filtered_array5)
    cv.imshow('3x3 gaussian filter PSNR:{}'.format(cv.PSNR(golden_image, gaussian_blur3)), gaussian_blur3)
    cv.imshow('5x5 gaussian filter PSNR:{}'.format(cv.PSNR(golden_image, gaussian_blur5)), gaussian_blur5)
    cv.imshow('adaptive mean filter PSNR:{}'.format(cv.PSNR(golden_image, adaptive_mean)), adaptive_mean)
    cv.imshow('Opencv bilateral filter PSNR:{}'.format(cv.PSNR(golden_image, bilateral_cv)), bilateral_cv)
    cv.imshow('My bilateral filter PSNR:{}'.format(cv.PSNR(golden_image, my_bilateral)), my_bilateral)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return bilateral_cv


def gaussian(x, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def distance(x1, y1, x2, y2):
    return np.sqrt(np.abs((x1 - x2) ** 2 + (y1 - y2) ** 2))


def bilateral_filter(image, diameter, sigmaColor, sigmaSpace):
    filtered_image = np.zeros(image.shape)
    padding_size = int((diameter - 1) / 2)
    image = np.pad(image, pad_width=padding_size, mode='edge')

    for i in range(padding_size, len(image) - padding_size):
        for j in range(padding_size, len(image) - padding_size):
            temp = 0
            wp_total = 0
            for k in range(i - padding_size, i + padding_size + 1):
                for l in range(j - padding_size, j + padding_size + 1):
                    gaussian_space = gaussian(distance(i, j, k, l), sigmaSpace)
                    gaussian_range = gaussian(np.abs(image[i][j] - image[k][l]), sigmaColor)
                    wp = gaussian_range * gaussian_space
                    temp += ((gaussian_range * gaussian_space) * image[k][l])
                    wp_total += wp
            filtered_image[i - padding_size][j - padding_size] = temp / wp_total

    filtered_image = cv.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                  dtype=cv.CV_32F)

    filtered_image = np.around(filtered_image)
    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image


def q3(image):
    my_bilateral_image = bilateral_filter(image, 3, 0.1, 1)
    opencv_bilateral = bilateralOpencv(image, 3, 0.1, 1)
    diff_array = cv.absdiff(my_bilateral_image, opencv_bilateral)
    print(np.max(diff_array[5:,5:]))
    cv.imshow('My bilateral Image PSNR:{}'.format(cv.PSNR(golden_image, my_bilateral_image)), my_bilateral_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # q1(noisyImage_gaussian)
    # q2(lower_gaussian_noise_image)
    q3(lower_gaussian_noise_image)