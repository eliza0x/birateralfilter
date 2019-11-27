import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils.extmath import cartesian
import math
from multiprocessing import Pool
import multiprocessing as multi
import itertools

def addNoise(img2D, mode, param):
    # % -> val
    param = param*255/100    # ノイズの生成
    noise = np.zeros(img2D.shape)
    # salt & pepper noise
    if mode == 's&p':
        tmp = np.zeros(img2D.shape)
        cv2.randu(tmp, 0, 255)
        noise[tmp<=param/2] = -255 # pepper
        noise[tmp>=255-param/2] = 255 # salt
    # white noise
    elif mode == 'white':
        cv2.randu(noise, -param/2, param/2)
    elif mode == 'gaussian':
        cv2.randn(noise, 0, param)
    else:
        print('有効なノイズが設定されていません')
    img2D = img2D + noise # クリッピング（範囲を超えた値を切って修正）
    img2D[img2D>255] = 255
    img2D[img2D<0] = 0    
    return img2D.astype(np.uint8)

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
img_noised = addNoise(img.copy(),'s&p',2)
d = 50

"""
@src: 入力画像
@d: 注目画素をぼかすために使われる領域
@sigmaColor: 色についての標準偏差
@sigmaSpace: 距離についての標準偏差
"""
def filtered(src, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)

def snr(img_orig, img):
    x, y = img_orig.shape
    img_orig = img_orig.astype(np.float32)
    img = img.astype(np.float32)

    s = np.square(img.sum())
    n = np.square((img - img_orig).sum())
    return 10 * math.log10(s/n)

def solve(p):
    (i, sigmaColor), (l, sigmaSpace) = p
    img_filtered = filtered(img_noised, d, sigmaColor, sigmaSpace)
    return (i, l), (sigmaColor, sigmaSpace), snr(img, img_filtered)

def plot(values):
    x = np.arange(0, 300, 10)
    y = np.arange(0, 300, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.empty(X.shape, dtype=np.float32)
    sigmaColors = np.empty(X.shape, dtype=np.float32)
    sigmaSpaces = np.empty(X.shape, dtype=np.float32)
    for pos, v, x in values:
        i, l = pos
        sigmaColors[i][l], sigmaSpaces[i][l] = v
        Z[i][l] = x
    fig = plt.figure() 
    ax = Axes3D(fig)
    ax.set_xlabel('sigmaColor')
    ax.set_ylabel('sigmaSpace')
    ax.set_zlabel('SNR')
    ax.set_title('BILATERAL FILTERED IMG SNR (d = '+str(d)+')')
    ax.plot_wireframe(X, Y, Z, color='blue',linewidth=0.3)
    plt.savefig('./graph/bilateralFilter(d='+str(d)+').png')
    bestIndex = np.unravel_index(np.argmax(Z), Z.shape)
    bestSigmaColor = sigmaColors[bestIndex[0]][bestIndex[1]]
    bestSigmaSpace = sigmaSpaces[bestIndex[0]][bestIndex[1]]
    print("BILATERAL FILTERED IMG MAX SNR (d="+str(d)+", sigmaColor="+str(bestSigmaColor)+", sigmaSpace="+str(bestSigmaSpace)+"): " + str(Z.max()))
    img_filtered = filtered(img_noised, d, bestSigmaColor, bestSigmaSpace)
    cv2.imwrite("./images/lena_gray_sp_bilateral_filtered(d="+str(d)+").png", img_filtered)
    # plt.show()

def main():
    global d
    cv2.imwrite("lena_gray.png", img)
    cv2.imwrite("lena_gray_sp.png", img_noised)
    print("NOISED IMG SNR: " + str(snr(img, img_noised)))
    for i in range(2, 10, 1):
        p = Pool(multi.cpu_count())
        values = p.map(solve, itertools.product(enumerate(range(0, 300, 10)), enumerate(range(0, 300, 10))))
        p.close()
        plot(values)
        d = 5 * i

if __name__ == '__main__':
    main()
