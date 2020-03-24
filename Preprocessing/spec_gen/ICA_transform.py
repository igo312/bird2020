# -*- coding:utf-8 -*-
# 有论文采用了取倒谱的方法，在多尺度变换下，则将输入固定，不需要网络结构的调整，可以参考。但是倒谱对人不是很友好。
# 更改图片的保存路径，一种鸟类存放所有图片， 不再是一段音频。 图片命名规则 ‘音频原文件名-{:index}’，文件名用来索取label，index用于保存
# TODO：怎么消除一段音频的静音段？
# TODO:多进程未完成
# TODO:(3/20)提高NFFT，直接截取高频信号。

from scipy import ndimage, interpolate
import scipy.io.wavfile as wave
from scipy.io import loadmat
from Preprocessing.spec_gen import sigproc
from Preprocessing.spec_gen.CEEMDAN_ICA import sigle_channel_ICA
from librosa.core import stft

import cv2
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime
import logging




def hasbird(spec, threshold=16):
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

    # DBUGB: show?
    # print thresh
    # plt.imshow('BIRD?', img)
    # plt.show()
    # TODO: the threshold is not confirmed
    isbird = True
    #print(thresh)
    #plt.imshow(spec)
    #plt.show()
    if thresh <= threshold:
        isbird = False

    return isbird

def GetMag(sig, rate, winlen, winstep, NFFT, fuc):
    '''获取输入音频的频谱图'''

    mag = stft(np.asfortranarray(sig), n_fft=NFFT, hop_length=int(winstep*rate), win_length=int(winlen*rate),window=fuc)
    if not isinstance(mag, np.ndarray):
        print('frames return None')
        return None
    # 习惯上我们将频谱值表现在y轴上，故旋转
    return mag

def concat(mags):
    mag = np.concatenate([
        np.expand_dims(mags[0], axis=2),
        np.expand_dims(mags[1], axis=2),
        np.expand_dims(mags[2], axis=2)
    ],axis=2)
    return mag

# 以下参数都直接来源于kahst's code
# NFFT 通常取2的幂次
def GetMultiMag(path, rate, second=5, overlap=3, minlen=3, winlen=0.05, winstep=0.0097, NFFT=1024, fuc='hann'):
    # 考虑物种只有10种，数据量不大，减少second生成更多spec

    # Read the audio
    sig = loadmat(path)

    # split into chunks with overlap
    sig = sig['component']
    sig_splits = []

    # overlap = second*0.5
    croplen = int(second * rate)
    for i in range(0, sig.shape[1], int((second - overlap) * rate)):
        # 分割之后将出现某些片段过短，这是不需要的故需要加入判断句
        split = sig[:,i:i + croplen]
        if split.shape[1] != croplen:
            # We will quit the short split before, but this time we save it with appending zero points.
            if len(split) < minlen*rate:
                continue
            else:
                zeros = np.zeros([split.shape[0], croplen-split.shape[1]])
                split = np.concatenate([split,zeros],axis=1)
        sig_splits.append(split)

    # Doing fourier transform

    # TODO:there is a question is about how to make the length in temporal axis is same to others?
    # multiprocess function

    for sigs in sig_splits:
        # 在获取频谱图之前我们需要进行预加重
        specs = []
        for sig in sigs:
            pre = sigproc.preemphasis(sig, coeff=0.95)

            # ICA 提取成分
            #if ICA:
            #   pre = ICA(pre, method='CEEMDAN')

            # 将buff转换为相应的频谱图

            MagSpec = GetMag(pre, rate, winlen, winstep, NFFT, fuc=fuc)
            if not isinstance(MagSpec, np.ndarray):
                continue
            # Get the power spec
            MagSpec = abs(MagSpec)
            # Normalize Process
            MagSpec -= MagSpec.min(axis=None)
            MagSpec /= MagSpec.max(axis=None)
            MagSpec = MagSpec[:512,:]
            # MagSpec = MagSpec[:256, :512]
            magspec = cv2.resize(MagSpec, (256, 256))
            specs.append(magspec)
        yield concat(specs)

def check(filename, strs):
    if filename in strs:
        return True
    else:
        return False

if __name__ == '__main__':
    # hyperparameter setting
    WinLen = 0.046
    WinStep = 0.01
    NFFT = 2048
    second = 5
    rate = 44100

    src_dir = r'G:\dataset\BirdClef\vacation\ica'
    spec_dir = r'G:\dataset\BirdClef\vacation\spectrums_total\ICAS1'
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir)

    matlists = os.listdir(src_dir)
    strs = ''.join(os.listdir(spec_dir))
    # 有些音频会因为太短而直接被放弃吧 LIFECLEF2017_BIRD_XC_WAV_RN49356.wav.mat， LIFECLEF2017_BIRD_XC_WAV_RN47572.wav.mat
    for fileindex, matname in enumerate(matlists):
        file_path = os.path.join(src_dir, matname)
        # Get the audio save path
        #wav_name = FileName.split('.')[0]
        #if wav_name in exist_list:
        #    continue
        if check(filename=matname, strs=strs):
            print('index-{} filename-{} already processed'.format(fileindex,matname))
            continue
        wav_info = GetMultiMag(file_path, rate=rate, second=second,winlen=WinLen, winstep=WinStep, NFFT=NFFT)
        noise_index = 0
        for index, img in enumerate(wav_info):
            has_bird=True
            # every spec_list
            if has_bird:
                spec_name = matname + '-{}'.format(index) + '.png'
                cv2.imwrite(os.path.join(spec_dir, spec_name), img * 255)
            else:
                noise_name_img = matname + 'noise-{}'.format(noise_index) + '.png'
                noise_index += 1
                cv2.imwrite(os.path.join(spec_dir, noise_name_img), img * 255)
        print('index-{} filename-{} process done'.format(fileindex, matname))
    print('Done.')