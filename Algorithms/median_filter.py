import cv2 as cv
import numpy as np


def q1(array=None):
    if array is None:
        image = cv.imread('../Images/noisyImage_SaltPepper.jpg', 0)
        image_array = np.asarray(image)
    else:
        image_array = array

    padding_size = 2

    array_with_padding = np.pad(image_array, pad_width=padding_size, mode='edge')

    filtered_array = np.zeros((len(image_array), len(image_array)))

    upper_range = len(array_with_padding) - padding_size
    for i in range(padding_size, upper_range):
        for j in range(padding_size, upper_range):
            med = np.median(
                array_with_padding[i - padding_size:i + padding_size + 1, j - padding_size: j + padding_size + 1])
            filtered_array[i - padding_size][j - padding_size] = med

    filtered_array = filtered_array.astype(np.uint8)

    if array is None:
        opencv_filtered_array = cv.medianBlur(image_array, 5)

        diff_array = cv.absdiff(filtered_array, opencv_filtered_array)
        print("MAX DIFF: ", np.max(diff_array))

        cv.imshow('opencv median filter', opencv_filtered_array)
        cv.imshow('my median filter', filtered_array)

        cv.waitKey(0)
        cv.destroyAllWindows()
    return filtered_array


def q2():
    image = cv.imread('../Images/noisyImage_SaltPepper.jpg', 0)
    golden_image = cv.imread('../Images/lena_grayscale_hq.jpg', 0)

    image_array = np.asarray(image)

    box_filtered_array = cv.blur(image_array, (5, 5), borderType=cv.BORDER_REPLICATE)
    gaussian_blur = cv.GaussianBlur(image_array, (7, 7), 0)
    opencv_filtered_array = cv.medianBlur(image_array, 5)

    psnr_box_filter = cv.PSNR(golden_image, box_filtered_array)
    psnr_gaussian = cv.PSNR(golden_image, gaussian_blur)
    psnr_median = cv.PSNR(golden_image, opencv_filtered_array)
    print("PSNR Box Filter", psnr_box_filter)
    print("PSNR Gaussian", psnr_gaussian)
    print("PSNR Median", psnr_median)


def q3(array=None):
    if array is None:
        image = cv.imread('../Images/noisyImage_SaltPepper.jpg', 0)

        image_array = np.asarray(image)
    else:
        image_array = array

    padding_size = 2

    array_with_padding = np.pad(image_array, pad_width=padding_size, mode='edge')

    filtered_array = np.zeros((len(image_array), len(image_array)))

    upper_range = len(array_with_padding) - padding_size
    for i in range(padding_size, upper_range):
        for j in range(padding_size, upper_range):
            flatten_array = array_with_padding[i - padding_size:i + padding_size + 1,
                            j - padding_size: j + padding_size + 1].flatten()
            med = np.median(np.sort(np.concatenate((flatten_array, np.array(
                [array_with_padding[i][j], array_with_padding[i][j]])))))
            filtered_array[i - padding_size][j - padding_size] = med

    filtered_array = filtered_array.astype(np.uint8)

    golden_image = cv.imread('../Images/lena_grayscale_hq.jpg', 0)

    golden_image_array = np.asarray(golden_image)

    if array is None:
        q1_median_filter = q1(image_array)
        box_filtered_array = cv.blur(image_array, (5, 5), borderType=cv.BORDER_REPLICATE)
        gaussian_blur = cv.GaussianBlur(image, (7, 7), 0)
        opencv_filtered_array = cv.medianBlur(image_array, 5)

        psnr_box_filter = cv.PSNR(golden_image_array, box_filtered_array)
        psnr_gaussian = cv.PSNR(golden_image_array, gaussian_blur)
        psnr_median = cv.PSNR(golden_image_array, opencv_filtered_array)
        psnr_my_median = cv.PSNR(golden_image_array, q1_median_filter)
        psnr_my_weighted_median = cv.PSNR(golden_image_array, filtered_array)
        print("PSNR MY Median", psnr_my_median)
        print("PSNR Box Filter", psnr_box_filter)
        print("PSNR Gaussian", psnr_gaussian)
        print("PSNR Median", psnr_median)
        print("PSNR My weighted median", psnr_my_weighted_median)

        cv.imshow('my center weighted median. PSNR: {}'.format(psnr_my_weighted_median), filtered_array)
        cv.imshow('gaussian. PSNR: {}'.format(psnr_gaussian), gaussian_blur)
        cv.imshow('box filter. PSNR: {}'.format(psnr_box_filter), box_filtered_array)
        cv.imshow('median. PSNR: {}'.format(psnr_my_median), opencv_filtered_array)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return filtered_array


def q4():
    img = cv.imread('../Images/noisyImage_SaltPepper.jpg', 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    dst = cv.warpAffine(img, M, (cols, rows))

    temp = q3(dst)
    golden_image = cv.imread('../Images/lena_grayscale_hq.jpg', 0)

    golden_image_array = np.asarray(golden_image)

    psnr_shifted = cv.PSNR(golden_image_array, temp)
    print("PSNR Shifted", psnr_shifted)
    cv.imshow('Good Looking Shifted Image with low PSNR: {}'.format(psnr_shifted), temp)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    print("QUESTION 1")
    q1()
    print("QUESTION 2")
    q2()
    print("QUESTION 3")
    q3()
    print("QUESTION 4")
    q4()
