import cv2 as cv
import numpy as np

noisyImage_gaussian = cv.imread('../Images/noisyImage_Gaussian.jpg', 0)
noisyImage_SaltPepper = cv.imread('../Images/noisyImage_SaltPepper.jpg', 0)
noisy_image = cv.imread('../Images/noisyImage.jpg', 0)
golden_image = cv.imread('../Images/lena_grayscale_hq.jpg', 0)


def q1():
    noise_variance = 0.004
    gaussian_array = noisyImage_gaussian
    denoised_array = np.zeros((len(gaussian_array), len(gaussian_array)))
    gaussian_array = cv.normalize(gaussian_array, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    array_with_padding = np.pad(gaussian_array, pad_width=2, mode='edge')
    for i in range(2, len(array_with_padding) - 2):
        for j in range(2, len(array_with_padding) - 2):
            Sxy = array_with_padding[i - 2:i + 3, j - 2:j + 3]
            denoised_array[i - 2][j - 2] = (array_with_padding[i][j] - ((
                    noise_variance / np.var(Sxy)) * (array_with_padding[i][j] - np.average(Sxy))))

    denoised_array = cv.normalize(denoised_array, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    denoised_array = denoised_array.astype(np.uint8)

    opencv_box_filtered_array = cv.blur(noisyImage_gaussian, (5, 5), borderType=cv.BORDER_REPLICATE)
    gaussian_blur = cv.GaussianBlur(noisyImage_gaussian, (5, 5), 0.9)

    psnr_box_filter = cv.PSNR(golden_image, opencv_box_filtered_array)
    psnr_gaussian_filter = cv.PSNR(golden_image, gaussian_blur)
    psnr_denoised_filter = cv.PSNR(golden_image, denoised_array)

    cv.imshow('output_1_1 | PSNR:{}'.format(psnr_denoised_filter), denoised_array)
    cv.imshow('output_1_2 | PSNR:{}'.format(psnr_box_filter), opencv_box_filtered_array)
    cv.imshow('output_1_3 | PSNR:{}'.format(psnr_gaussian_filter), gaussian_blur)
    cv.waitKey(0)
    cv.destroyAllWindows()


def q2():
    padding_sizes = [1, 2, 3]
    salt_and_pepper_array = noisyImage_gaussian
    denoised_array = np.zeros((len(salt_and_pepper_array), len(salt_and_pepper_array)))
    salt_and_pepper_array = cv.normalize(salt_and_pepper_array, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,
                                         dtype=cv.CV_32F)
    salt_and_pepper_array = np.pad(salt_and_pepper_array, pad_width=3, mode='edge')

    for i in range(3, len(salt_and_pepper_array) - 3):
        for j in range(3, len(salt_and_pepper_array) - 3):
            for k in range(0, len(padding_sizes)):
                arr = salt_and_pepper_array[i - padding_sizes[k]:i + padding_sizes[k] + 1,
                       j - padding_sizes[k]: j + padding_sizes[k] + 1]
                zmin = np.min(salt_and_pepper_array[i - padding_sizes[k]:i + padding_sizes[k] + 1,
                       j - padding_sizes[k]: j + padding_sizes[k] + 1])
                zmed = np.median(salt_and_pepper_array[i - padding_sizes[k]:i + padding_sizes[k] + 1,
                       j - padding_sizes[k]: j + padding_sizes[k] + 1])
                zmax = np.max(salt_and_pepper_array[i - padding_sizes[k]:i + padding_sizes[k] + 1,
                       j - padding_sizes[k]: j + padding_sizes[k] + 1])
                if zmin < zmed < zmax:
                    if zmin < salt_and_pepper_array[i][j] < zmax:
                        denoised_array[i - 3][j - 3] = salt_and_pepper_array[i][j]
                        break
                    else:
                        denoised_array[i - 3][j - 3] = zmed
                        break
                else:
                    if k == 3:
                        denoised_array[i-3][j-3] = zmed

    denoised_array = cv.normalize(denoised_array, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    denoised_array = denoised_array.astype(np.uint8)
    opencv_filtered_array3 = cv.medianBlur(noisyImage_SaltPepper, 3)
    opencv_filtered_array5 = cv.medianBlur(noisyImage_SaltPepper, 5)
    opencv_filtered_array7 = cv.medianBlur(noisyImage_SaltPepper, 7)

    center_weighted_3 = weighted_median(filterSize=3, centerWeight=3)
    center_weighted_5 = weighted_median(5,5)
    center_weighted_7 = weighted_median(7,7)

    psnr_denoised_array = cv.PSNR(golden_image, denoised_array)
    psnr_opencv_filtered_3 = cv.PSNR(golden_image, opencv_filtered_array3)
    psnr_opencv_filtered_5 = cv.PSNR(golden_image, opencv_filtered_array5)
    psnr_opencv_filtered_7 = cv.PSNR(golden_image, opencv_filtered_array7)
    psnr_center_weighted_3 = cv.PSNR(golden_image, center_weighted_3)
    psnr_center_weighted_5 = cv.PSNR(golden_image, center_weighted_5)
    psnr_center_weighted_7 = cv.PSNR(golden_image, center_weighted_7)

    cv.imshow('output_2_1 | PSNR:{}'.format(psnr_denoised_array), denoised_array)
    cv.imshow('output_2_2 | PSNR:{}'.format(psnr_opencv_filtered_3), opencv_filtered_array3)
    cv.imshow('output_2_3 | PSNR:{}'.format(psnr_opencv_filtered_5), opencv_filtered_array5)
    cv.imshow('output_2_4 | PSNR:{}'.format(psnr_opencv_filtered_7), opencv_filtered_array7)
    cv.imshow('output_2_5 | PSNR:{}'.format(psnr_center_weighted_3), center_weighted_3)
    cv.imshow('output_2_6 | PSNR:{}'.format(psnr_center_weighted_5), center_weighted_5)
    cv.imshow('output_2_7 | PSNR:{}'.format(psnr_center_weighted_7), center_weighted_7)
    cv.waitKey(0)
    cv.destroyAllWindows()


def weighted_median(filterSize, centerWeight):
    padding_size = int((filterSize - 1) / 2)

    array_with_padding = np.pad(noisyImage_SaltPepper, pad_width=padding_size, mode='edge')

    filtered_array = np.zeros((len(noisyImage_SaltPepper), len(noisyImage_SaltPepper)))

    upper_range = len(array_with_padding) - padding_size
    for i in range(padding_size, upper_range):
        for j in range(padding_size, upper_range):
            flatten_array = array_with_padding[i - padding_size:i + padding_size + 1,
                            j - padding_size: j + padding_size + 1].flatten()
            med = np.median(np.sort(np.concatenate((flatten_array, np.array(
                ([array_with_padding[i][j]] * (centerWeight-1)))))))
            filtered_array[i - padding_size][j - padding_size] = med

    filtered_array = filtered_array.astype(np.uint8)
    return filtered_array

if __name__ == '__main__':
    print("QUESTION 1")
    q1()
    # print("QUESTION 2")
    # q2()
