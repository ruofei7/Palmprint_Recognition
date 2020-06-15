import matplotlib.pyplot as plt  # plt 用于显示图片
from matplotlib.pyplot import imread
from skimage import morphology
import numpy as np
import glob
import os
import cv2


# %%
class Palm_Graph():
    def __init__(self, train, test):
        assert (np.array(train).shape == np.array(test).shape)
        self.train = train
        self.test = test
        self.rows, self.cols = train[0].shape


def Low_pass_Gausian_process(img, D0):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    Locx = np.array(list(range(cols)) * rows).reshape([rows, cols])
    Locy = np.transpose((np.array(list(range(rows)) * cols).reshape([cols, rows])))
    D = np.sqrt((Locx - cols / 2) ** 2 + (Locy - rows / 2) ** 2)
    mask = np.exp(-D ** 2 / D0 ** 2 / 2)

    f1 = np.fft.fft2(img)
    f1shift = np.fft.fftshift(f1)
    f1shift = f1shift * mask
    f2shift = np.fft.ifftshift(f1shift)
    img_new = np.fft.ifft2(f2shift)
    img_new = np.abs(img_new)

    return img_new


class gabor():
    def __init__(self, R, C, n_orientation, scale):
        self.R = R
        self.C = C
        self.n_orientarion = n_orientation
        self.scale = scale
        self.orientation = np.array([u * np.pi / n_orientation for u in range(1, n_orientation + 1)])
        self.gabor_filters_sets = [gabor_wavelet(R, C, u, scale, n_orientation) for u in range(1, n_orientation + 1)]

    def filtering(self, img):
        # 返回n_orientarion个滤波后的图像
        graphs = np.array([cv2.filter2D(img, -1, np.real(gw)) for gw in self.gabor_filters_sets])
        return graphs

    def plot_filters(self, n_scale):
        gabor_filters = []
        fig = plt.figure()
        for v in range(1, n_scale + 1):
            for u in range(1, self.n_orientarion + 1):
                gw = gabor_wavelet(self.R, self.C, u, v, self.n_orientarion)
                fig.add_subplot(n_scale, self.n_orientarion, self.n_orientarion * (v - 1) + u)
                plt.imshow(np.real(gw), cmap='gray')
        plt.show()


def gabor_wavelet(rows, cols, orientation, scale, n_orientation):
    kmax = np.pi / 2
    f = np.sqrt(2)
    delt2 = (2 * np.pi) ** 2
    k = (kmax / (f ** scale)) * np.exp(1j * orientation * np.pi / n_orientation / 2)
    kn2 = np.abs(k) ** 2
    gw = np.zeros((rows, cols), np.complex128)

    for m in range(int(-rows / 2) + 1, int(rows / 2) + 1):
        for n in range(int(-cols / 2) + 1, int(cols / 2) + 1):
            t1 = np.exp(-0.5 * kn2 * (m ** 2 + n ** 2) / delt2)
            t2 = np.exp(1j * (np.real(k) * m + np.imag(k) * n))
            t3 = np.exp(-0.5 * delt2)
            gw[int(m + rows / 2 - 1), int(n + cols / 2 - 1)] = (kn2 / delt2) * t1 * (t2 - t3)

    return gw


def get_data(number):
    number = str(number)
    train_files = sorted(glob.glob(os.path.join(train_data_path, number.zfill(3) + '*.bmp')))
    test_files = sorted(glob.glob(os.path.join(test_data_path, number.zfill(3) + '*.bmp')))
    train_data = [imread(graph) for graph in train_files]
    test_data = [imread(graph) for graph in test_files]
    palm = Palm_Graph(train_data, test_data)
    return palm


def LOG_preprocess(img, R0=40, ksize=5):
    AfterGaussian = np.uint8(Low_pass_Gausian_process(img, R0))
    processed = cv2.Laplacian(AfterGaussian, -1, ksize=ksize)
    img = cv2.equalizeHist(img)
    return processed
def process(img):
    img = LOG_preprocess(img)  # 预处理，高斯+laplacian

    After_gabor = []
    #     fig = plt.figure(dpi=150)
    for i, gw in enumerate(gabor_filters):
        element = cv2.filter2D(img, -1, np.real(gw))
        After_gabor.append(element)

    Two_value = []
    #     fig = plt.figure(dpi=150)
    for i, line in enumerate(After_gabor):
        _, TW = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        TW = cv2.erode(TW, kernel)

        Two_value.append(TW)
    con = []
    for i in Two_value:
        conective = morphology.remove_small_objects(i > 0, min_size=40, connectivity=1, in_place=False)
        con.append(conective)

    line = (np.sum(con, axis=0) / len(con))

    return line

train_data_path='./Palmprint/training'
test_data_path='./Palmprint/testing'
n_orientation=6
scale=2
GA = gabor(10,10,n_orientation,scale) #10*10的filter,取6个不同角度,尺度为2
gabor_filters = GA.gabor_filters_sets
GA.plot_filters(3) # 打印n种尺度的滤波器

palm1=get_data(1)
palm2=get_data(2)
train1=palm1.train[0]
test1 = palm1.test[0]
train2=palm2.train[0]
test2 = palm2.test[0]

plt.imshow(train2, cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(LOG_preprocess(train2), cmap='gray')
plt.axis('off')
plt.show()

res1 = process(train2)
fig = plt.figure(dpi=70)
plt.imshow(res1, cmap='gray')
plt.axis('off')
plt.show()