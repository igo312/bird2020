import numpy as np
from scipy import ndimage
import cv2

import matplotlib.pyplot as plt
import os
import shutil

def hasbird(spec, ht=120, lt=16):
    '''
            判断输入的音频是不是含有鸟声
            步骤：
            1.median blur
            2.get median threshold and where less than the values fix to zero
            3.spot removal
            4.morphological closing
            5.judge
        主要的判别方法是先采用close再dialation获取非零的像素点与threshold比较来判断
    '''

    def filter_isolated_cell(magspec, struct):
        '''利用ndimage的label进行对于孤立点的过滤'''

        # 获取到每个位置对应的编码
        id_regions, numfeature = ndimage.label(magspec, struct)

        # 对于每个label所占有的数量的进行求和
        id_sizes = ndimage.sum(magspec, id_regions, range(numfeature + 1))

        # 获取size=1的label并将其删除
        area_mask = (id_sizes == 1)
        magspec[area_mask[id_regions]] = 0

        # 或者可以采用
        # id_iso = np.where(id_sizes==1)[0]
        # for i in id_iso: magspec[id_regions==i] = 0

        return magspec
    # working copy
    imgs = spec.copy()

    isbird = True
    # the sigma is more less, the spec will be more smooth
    for i in range(3):
        img = imgs[:,:,i]
        img = cv2.medianBlur(img, 5)

        # STEP 2: median thereshold calculate
        # keepdims = True 使得在比较中，raw_median只会跟raw比， col_median只会跟col比
        col_median = np.median(img, axis=0, keepdims=True)
        raw_median = np.median(img, axis=1, keepdims=True)

        #
        img[img < col_median * 3] = 0
        img[img < raw_median * 4] = 0
        img[img > 0] = 1

        # STEP 3: remove isolated signal
        img = filter_isolated_cell(img, struct=np.ones((3, 3)))

        # STEP 4: Morph Closing
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.float32))

        # STEP 5: 利用图形学计算threshold并与给定的threshold比较

        # 避免过大过小频率影响，选择频率
        img = img[128:-16, :]

        # 计算cthresh
        col_max = np.max(img, axis=0)
        col_max = ndimage.morphology.binary_dilation(col_max, iterations=2).astype(
            col_max.dtype)  # struct 将才用default 图形，即一个简单十字
        cthresh = np.sum(col_max)

        # 计算rthresh
        raw_max = np.max(img, axis=1)
        raw_max = ndimage.morphology.binary_dilation(raw_max, iterations=2).astype(col_max.dtype)
        rthresh = np.sum(raw_max)

        # 这里将与给定threshold进行比较。比较方式多样，但源代码中只是将rthresh作为比较
        # final thresh
        thresh = rthresh

        # print thresh
        # plt.imshow('BIRD?', img)
        # plt.show()
        # TODO: the threshold is not confirmed
        isbird = True
        print(thresh)
        #plt.imshow(spec)
        #plt.show()
        if lt< thresh< ht:
            isbird = True
    print('-------------------\n')
    print('-------------------\n')
    return isbird

if __name__ == '__main__':
    spec_dir = r'G:\dataset\BirdClef\vacation\mat'
    save_dir = r'G:\dataset\BirdClef\vacation\mat_hasbird'
    specs = os.listdir(spec_dir)
    for filename in specs:
        path = os.path.join(spec_dir, filename)
        img = plt.imread(path)
        ht = 200
        lt = 16
        hasbird(img, ht, lt)
        #if hasbird(img, ht, lt):
        #    dst = os.path.join(save_dir, filename)
        #    shutil.copy(path, dst)

