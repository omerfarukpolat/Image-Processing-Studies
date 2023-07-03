import cv2 as cv
import numpy as np
import random

equalized_image = cv.imread('../Images/mytest.jpg', 0)


# Interpolation function
def interpolation(x, y, image, centers, index, tile_size, original_image):
    rightTopIndex = index + 1
    rightBottomIndex = int(len(image[0]) / tile_size + index + 1)
    leftTopIndex = index
    leftBottomIndex = int(len(image[0]) / tile_size + index)
    x1 = centers[index][0]
    y1 = centers[index][1]
    leftTopIntensity = centers[leftTopIndex][2][original_image[x][y]]
    leftBottomIntensity = centers[leftBottomIndex][2][original_image[x][y]]
    rightTopIntensity = centers[rightTopIndex][2][original_image[x][y]]
    rightBottomIntensity = centers[rightBottomIndex][2][original_image[x][y]]
    rPrime = tile_size - (x - x1)
    cPrime = tile_size - (y - y1)

    return np.round(((rightBottomIntensity * (tile_size - cPrime) * (tile_size - rPrime)) +
                    (leftTopIntensity * rPrime * cPrime) +
                    (leftBottomIntensity * (tile_size - rPrime) * cPrime) +
                    (rightTopIntensity * rPrime * (tile_size - cPrime))) / tile_size ** 2)

def efficient_clahe(image, tile_size, clip):
    height, width = image.shape
    remainderWidth = (tile_size * (int(width / tile_size) + 1)) - width
    remainderHeight = (tile_size * (int(height / tile_size) + 1)) - height
    padded_image = cv.copyMakeBorder(image, 0, remainderHeight, 0, remainderWidth, borderType=cv.BORDER_REFLECT)
    first_image = cv.copyMakeBorder(image, 0, remainderHeight, 0, remainderWidth, borderType=cv.BORDER_REFLECT)

    height, width = padded_image.shape
    centers = []
    centersIndex = 0
    halfTile = int(tile_size / 2) - 1
    for i in range(halfTile, height - halfTile, tile_size):
        for j in range(halfTile, width - halfTile, tile_size):
            centers.append([i, j])

    for (x, y) in centers:
        neighbourhood = padded_image[x - halfTile: x + halfTile + 2, y - halfTile: y + halfTile + 2]
        num_pixels = len(neighbourhood.flatten())
        histogram = np.zeros(num_pixels).astype(np.uint8)
        neighbourhood_flatten = neighbourhood.flatten()
        for i in range(0, num_pixels):
            histogram[neighbourhood_flatten[i]] += 1
        while np.max(histogram) > clip:
            for j in range(0, num_pixels):
                if histogram[j] > clip:
                    diff = int(histogram[j] - clip)
                    histogram[j] -= diff
                    for k in range(0, diff):
                        min_value = random.randint(0, 255)
                        histogram[min_value] += 1

        histogram = histogram / num_pixels
        cdf_histogram = np.cumsum(histogram)
        lookup_table = np.round_(255 * cdf_histogram).astype(np.uint8)
        centers[centersIndex].append(lookup_table)

        for i in range(0, len(neighbourhood)):
            for j in range(0, len(neighbourhood)):
                neighbourhood[i][j] = lookup_table[neighbourhood[i][j]]

        padded_image[x - halfTile: x + halfTile + 2, y - halfTile: y + halfTile + 2] = neighbourhood
        centersIndex += 1
    for k in range(0, len(centers)):
        x1 = centers[k][0]
        y1 = centers[k][1]
        if y1 + tile_size < width and x1 + tile_size < height:
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            for i in range(x1, x2):
                for j in range(y1, y2):
                    padded_image[i][j] = interpolation(i, j, padded_image, centers, k, tile_size, first_image)


    res = np.hstack((image, padded_image[0:len(image), 0:len(image[0])]))
    cv.imshow("img", res)
    cv.waitKey(0)


if __name__ == '__main__':
    efficient_clahe(equalized_image, 16, 4)
