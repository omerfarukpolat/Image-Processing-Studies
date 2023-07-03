import sys

import cv2 as cv
import numpy as np

def q1():
    image = cv.imread('../Images/lena_grayscale_hq.jpg', 0)
    image_array = np.asarray(image)
    filter_size_array = sys.argv

    output_counter = 1
    for p in range(1, len(filter_size_array)):
        filter_size = int(filter_size_array[p])
        padding_size = np.floor((filter_size - 1) / 2).astype(np.uint8)

        opencv_blurred_image = cv.blur(image, (int(filter_size), int(filter_size)), borderType=0)

        array_with_padding = cv.copyMakeBorder(image_array, padding_size, padding_size, padding_size, padding_size,
                                               borderType=0)

        blurred_image_array = np.zeros((len(image_array), len(image_array)))
        upper_range = len(array_with_padding) - padding_size
        box_size = filter_size * filter_size
        for i in range(padding_size, upper_range):
            for j in range(padding_size, upper_range):
                sum = np.sum(
                    array_with_padding[i - padding_size: i + padding_size + 1, j - padding_size: j + padding_size + 1])
                blurred_image_array[i - padding_size][j - padding_size] = int(sum / box_size)
        blurred_image_array = blurred_image_array.astype(np.uint8)

        diff_array = cv.absdiff(blurred_image_array, opencv_blurred_image)
        print("MAX DIFF FOR FILTER SIZE ", filter_size, ":", np.max(diff_array))

        cv.imshow('output_1_{}'.format(output_counter), blurred_image_array)
        cv.imshow('output_2_{}'.format(output_counter), opencv_blurred_image)
        output_counter += 1
        cv.imshow('difference_image for filter size: {} (check terminal for max difference)'.format(filter_size),
                  diff_array)
        cv.waitKey(0)
        cv.destroyAllWindows()


def q2():

    image = cv.imread('../Images/lena_grayscale_hq.jpg', 0)

    filter_size_array = sys.argv
    output_counter = 1
    for i in range(1, len(filter_size_array)):
        filter_size = int(filter_size_array[i])
        padding_size = int((filter_size - 1) / 2)

        image_array = np.asarray(image)

        padding_array = np.pad(image_array, pad_width=padding_size)
        intermediate_image = np.zeros((len(padding_array), len(padding_array)))
        final_array =  np.zeros((len(image_array), len(image_array)))

        intermediate_image = np.asarray(intermediate_image)
        final_array = np.asarray(final_array)
        for i in range(padding_size, len(image_array) + padding_size):
            for j in range(padding_size, len(image_array) + padding_size):
                total = 0
                for k in range(padding_size, -1 * padding_size - 1, -1):
                    total = total + int(padding_array[i - k][j])
                intermediate_image[i][j] = np.round(total / filter_size)

        for i in range(padding_size, len(image_array) + padding_size):
            for j in range(padding_size, len(image_array) + padding_size):
                total = 0
                for k in range(padding_size, -1 * padding_size - 1, -1):
                    total = total + int(intermediate_image[i][j - k])
                final_array[i - padding_size][j - padding_size] = np.round(total / filter_size)

        opencv_blurred_array = cv.blur(image, (int(filter_size), int(filter_size)), borderType=0)
        diff_array = cv.absdiff(final_array.astype(np.uint8), opencv_blurred_array)
        print("MAX DIFF FOR FILTER SIZE ", filter_size, ":", np.max(diff_array))

        cv.imshow('output_3_{}'.format(output_counter), opencv_blurred_array)
        cv.imshow('output_2_{}'.format(output_counter),
                  final_array.astype(np.uint8))
        output_counter+=1
        cv.imshow('difference_image for filter size: {} (check terminal for max difference)'.format(filter_size),
                  diff_array)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    print("QUESTION 1")
    q1()
    print("QUESTION 2")
    q2()
