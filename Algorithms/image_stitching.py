import sys

import cv2 as cv
import imutils
import numpy as np

img1 = cv.imread("../Images/uni_test_2.jpg")
img2 = cv.imread("../Images/uni_test_1.jpg")
img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)


def detectAndDescribe(image, method=None):
    if method == 'sift':
        descriptor = cv.xfeatures2d.SIFT_create()
    elif method == 'brisk':
        descriptor = cv.BRISK_create()
    elif method == 'orb':
        descriptor = cv.ORB_create()

    (kps, features) = descriptor.detectAndCompute(image, None)

    return kps, features


def createMatcher(method, crossCheck):
    if method == 'sift':
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
    best_matches = bf.match(featuresA, featuresB)

    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    return bf.match(featuresA, featuresB)


def matchKeyPointsKNN(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=False)
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m, n in rawMatches:
        if m.distance < 0.7*n.distance:
            matches.append(m)
    return matches

def matchKeyPointsFlann():
    detector = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matchess = flann.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for m, n in matchess:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, status = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    return good, H, status


def getHomography(kpsA, kpsB, matches, reprojThresh):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        (H1, status1) = cv.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh)

        return matches, H1, status1
    else:
        return None



if __name__ == '__main__':
    while(True):
        extractorChoice = int(input("1 FOR SIFT\n2 FOR BRISK\n3 FOR ORB\n0 FOR EXIT\n"))
        if extractorChoice == 1:
            feature_extractor = "sift"
        elif extractorChoice == 2:
            feature_extractor = "brisk"
        elif extractorChoice == 3:
            feature_extractor = "orb"
        else:
            sys.exit()
        matcherChoice = int(input("\n1 FOR BRUTE FORCE\n2 FOR FLANN\n"))
        if matcherChoice == 1:
            feature_matching = "bf"
        elif matcherChoice == 2:
            feature_matching = "knn"
        else:
            print("Wrong Input")

        kpsA, featuresA = detectAndDescribe(img1_gray, method=feature_extractor)
        kpsB, featuresB = detectAndDescribe(img2_gray, method=feature_extractor)

        if feature_matching == 'bf':
            matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        elif feature_matching == 'knn':
            matchess = matchKeyPointsFlann()

        if feature_matching == 'knn':
            M = matchess
        else:
            M = getHomography(kpsA, kpsB, matches, reprojThresh=4)
        (matches, H, status) = M

        width = img2.shape[1] + img1.shape[1]
        height = img2.shape[0] + img1.shape[0]

        result = cv.warpPerspective(img1, H, (width, height))

        result[0:img2.shape[0], 0:img2.shape[1]] = img2

        gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv.contourArea)
        (x, y, w, h) = cv.boundingRect(c)
        result = result[y:y + h, x:x + w]

        result = cv.resize(result, (1600, 900))
        cv.imshow("Stitched", result)
        cv.waitKey(0)
        cv.destroyAllWindows()

