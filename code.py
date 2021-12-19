import cv2
import numpy as np


img = cv2.imread('input.jpg', 0)
ip_xlen = len(img)
ip_ylen = len(img[0])
ip_size = ip_xlen * ip_ylen


def getHist(img):
    hist = np.zeros(256)
    for i in range(ip_xlen):
        for j in range(ip_ylen):
            hist[img[i][j]] += 1
    hist /= ip_size
    return hist


def getCdf(hist):
    H = np.zeros(256)
    add = 0
    for i in range(256):
        H[i] = add + hist[i]
        add += hist[i]
    return H


def equalize(img, H):
    mapping = np.zeros(256, int)
    out_img = np.zeros([ip_xlen, ip_ylen], int)
    for i in range(256):
        mapping[i] = int(round(255 * H[i]))
    for i in range(ip_xlen):
        for j in range(ip_ylen):
            out_img[i][j] = mapping[img[i][j]]
    return out_img


def gammaTrans(img, g):
    target = np.zeros([ip_xlen, ip_ylen], int)
    const = 255 / pow(255, g)
    for i in range(ip_xlen):
        for j in range(ip_ylen):
            target[i][j] = int(round(const * pow(img[i][j], g)))
    return target


def matchHist(H, G):
    mapping = np.zeros(256, int)
    out_img = np.zeros([ip_xlen, ip_ylen], int)
    pos = 0
    for i in range(256):
        while H[i] > G[pos]:
            pos += 1

        if H[i] == G[pos] or pos == 0:
            mapping[i] = pos
        else:
            if H[i] - G[pos - 1] > G[pos] - H[i]:
                mapping[i] = pos
            else:
                mapping[i] = pos - 1
    for i in range(ip_xlen):
        for j in range(ip_ylen):
            out_img[i][j] = int(mapping[img[i][j]])
    return out_img


def convolution(ip, filt):
    out = np.zeros([5, 5], int)
    inv_filt = np.flip(np.flip(filt, axis=0), axis=1)
    outOrigin = [1, 1]
    filtOrigin = [1, 1]
    for i in range(5):
        for j in range(5):
            for x in range(3):
                for y in range(3):
                    if i - outOrigin[0] + (x - filtOrigin[0]) >= 0 and i - outOrigin[0] + (x - filtOrigin[0]) < 3 and j - outOrigin[1] + (y - filtOrigin[1]) >= 0 and j - outOrigin[1] + (y - filtOrigin[1]) < 3:
                        out[i][j] += ip[i - outOrigin[0] + (x - filtOrigin[0])
                                        ][j - outOrigin[1] + (y - filtOrigin[1])] * inv_filt[x][y]
    return out


"""
ip_hist = getHist(img)

ip_H = getCdf(ip_hist)

out_img = equalize(img, ip_H)
op_hist = getHist(out_img)

print("\nNormalized histogram of the input image: ")
print(ip_hist,"\n")
print("\nNormalized historgram of the output image: ")
print(op_hist,"\n")
"""
"""
ip_hist = getHist(img)

ip_H = getCdf(ip_hist)

target = gammaTrans(img, 0.5)

target_hist = getHist(target)

target_H = getCdf(target_hist)

out_img = matchHist(ip_H, target_H)

op_hist = getHist(out_img)

print("\nNormalized histogram of the input image: ")
print(ip_hist, "\n")
print("\nNormalized histogram of the target image: ")
print(target_hist, "\n")
print("\nNormalized histogram of the output image: ")
print(op_hist, "\n")

cv2.imshow("input", img)
cv2.imshow("target", target / 255)
cv2.imshow("output", out_img / 255)
"""

ip = np.random.randint(low=1, high=10, size=9).reshape([3, 3])
filt = np.random.randint(low=1, high=10, size=9).reshape([3, 3])

#ip = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
#filt = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

out = convolution(ip, filt)
print("\nInput\n", ip)
print("\nFilter\n", filt)
print("\nOutput\n", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
