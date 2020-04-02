import numpy as np
from scipy import ndimage
import cv2

import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
def hasbird(spec, ht=120, lt=35):
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
    isbird = True

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
    img = spec.copy()


    # the sigma is more less, the spec will be more smooth

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
    # print(thresh)
    #plt.imshow(spec)
    #plt.show()
    if thresh>ht or thresh<lt:
        isbird = False
    #print('-------------------\n')
    #print('-------------------\n')
    return thresh, isbird

if __name__ == '__main__':
    path1 = r'G:\dataset\BirdClef\vacation\spectrums_total\ICAS1'
    path2 = r'G:\dataset\BirdClef\vacation\spectrums_total\pure'

    noise_dir = r'G:\dataset\BirdClef\vacation\spectrums_total\pure_hasbird_noise'
    save_dir = r'G:\dataset\BirdClef\vacation\spectrums_total\pure_hasbird'
    specs = os.listdir(path2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(noise_dir):
        os.makedirs(noise_dir)

    noise_index = 0
    for filename in specs:
        path = os.path.join(path2, filename)
        img = plt.imread(path)
        ht = 500
        lt = 35
        if len(img.shape) == 3:
            for i in range(3):
                thresh, has = hasbird(img[:,:,i], ht, lt)
                if thresh > 230:
                    print('{} thresh is {}'.format(filename, thresh))
                if has:
                    F = os.path.join(save_dir, filename.split('.png')[0]+'-{}.png'.format(i))
                    cv2.imwrite(F, img[:,:,i]*255)
                else:
                    F = os.path.join(noise_dir, filename.split('.png')[0]+'-{}.png'.format(noise_index))
                    noise_index += 1
                    cv2.imwrite(F, img[:,:,i]*255)
        else:
            thresh, has = hasbird(img, ht, lt)
            if thresh > 230:
                print('{} thresh is {}'.format(filename, thresh))
            if has:
                F = os.path.join(save_dir, filename)
                cv2.imwrite(F, img*255)
            else:
                F = os.path.join(noise_dir, filename.split('.png')[0] + '-{}.png'.format(noise_index))
                noise_index += 1
                cv2.imwrite(F, img*255)
        #if hasbird(img, ht, lt):
        #    dst = os.path.join(save_dir, filename)
        #    shutil.copy(path, dst)

