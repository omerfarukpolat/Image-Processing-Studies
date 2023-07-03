import cv2 as cv
import numpy as np

image = cv.imread('../Images/lena_grayscale_hq.jpg', 0)


def q1():
    myImage = image

    myImage = np.pad(myImage, (1, 0), mode='constant')

    image_array = np.asarray(myImage)

    integral_image = np.zeros((len(image_array), len(image_array)))

    for i in range(0, len(image_array)):
        for j in range(0, len(image_array)):
            integral_image[i][j] = np.sum(image_array[0:i + 1, 0:j + 1])

    integral_image = integral_image.astype(np.uint8)


    opencv_integral_image, tmp = cv.integral2(image)

    opencv_integral_image = opencv_integral_image.astype(np.uint8)

    difference_image = (integral_image - opencv_integral_image) * 100

    difference_image = difference_image[1:, 1:]
    difference_image = np.asarray(difference_image).astype(np.uint8)

    print("MAX DIFFERENCE BETWEEN MY INTEGRAL IMAGE AND OPENCV INTEGRAL IMAGE: ", np.max(difference_image))

    cv.imshow("difference image. Max difference: {}".format(np.max(difference_image)), difference_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def q2():
    opencv_integral_image, tmp = cv.integral2(image)

    opencv_integral_image = cv.copyMakeBorder(opencv_integral_image, 1,1,1,1, cv.BORDER_REPLICATE)

    leveraged_array = np.zeros((512, 512))

    for i in range(2, len(opencv_integral_image) - 1):
        for j in range(2, len(opencv_integral_image) - 1):
            leveraged_array[i-2][j-2] = np.round((opencv_integral_image[i + 1][j + 1] + opencv_integral_image[i - 2][j - 2] -
                                             opencv_integral_image[i + 1][j - 2] - opencv_integral_image[i - 2][j + 1] ) / 9)


    box_filtered_array = cv.blur(image, (3, 3), borderType= 0)

    leveraged_array = leveraged_array.astype(np.uint8)

    difference_image = cv.absdiff(leveraged_array, box_filtered_array)
    print("Max value of difference array: ", np.max(difference_image))
    cv.imshow('Leveraged Image. Difference: {}'.format(np.sum(difference_image)), leveraged_array)
    cv.imshow('Opencv 3x3 Box Filter Output', box_filtered_array)
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == '__main__':
    print("QUESTION 1 IS RUNNING CURRENTLY")
    q1()
    print("QUESTION 2 IS RUNNING CURRENTLY")
    q2()