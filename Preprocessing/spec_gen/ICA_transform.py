# -*- encoding:utf-8 -*-

# single wav to complete ICA:截取一段音频，以音频中点为均值正态分布选择起点，在这一段音频中再截取若干个窗口当作多个麦克风

from scipy import ndimage, interpolate
import python_speech_features as psf
from Preprocessing.spec_gen import sigproc
import scipy.io.wavfile as wave
from sklearn.decomposition import FastICA

import cv2
import pandas as pd

import random
import numpy as np
import os

from datetime import datetime

# cd频率，与梅尔刻度相对应
RATE = 44100

def ChangeSample(sig, rate):
    # 做一个频率的判断，减少运算过程
    if rate == RATE:
        return sig
    else:
        '''
        # 改变频率的运用：
        # 1.得到音频时间长度，根据 采样频率*时间 = 采样数，获得相应的音频
        # 2.在利用np.interpolate 差值函数即可
        '''
        # sig 是一个一维矩阵
        duration = sig.shape[0] / rate

        # 获得矩阵上的采样点位置
        old = np.linspace(0, duration, sig.shape[0])
        # int((sig.shape[0] / rate) * RATE采样点
        new = np.linspace(0, duration, int((sig.shape[0] / rate) * RATE))

        # 获得插值工具
        # https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
        inter = interpolate.interp1d(x=old, y=sig.T)  # x为采样点位置， y为相应数据，即在相应数据的位置上插值
        # TODO: 为什么要转置呢？
        new_sig = inter(new).T  # 这样即在新的采样位置上插值

        # 保证数据为整数方便计算
        new_sig = np.round(new_sig).astype(sig.dtype)
        sig = new_sig
    return sig

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

def single_to_multi(signal, target_channel=5):
    if len(signal.shape) == 1:
        # Expand channel dimension
        signal = np.expand_dims(signal, axis=0)

    for index in range(0, signal.shape-1, int((signal.shape-1)/(target_channel-1))):
        signal = np.concatenate([signal,
                                 np.expand_dims(np.append(signal[index:], signal[:index]), axis=0)],
                                axis=0)
    return signal

def GetMag(sigs, rate, winlen, winstep, NFFT, fuc_name='Rect'):
    '''获取输入音频的频谱图'''
    # 采用了矩形窗
    winfuc = {
              'Rect': lambda x: np.ones((x,)),
              'Hamming': lambda x: np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (x - 1)) for n in range(x)]),
              'Hanning': lambda x: np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (x - 1)) for n in range(x)])
              }

    # 窗函数
    # 得到一个frames，姑且认为是采样位置。函数返回一个矩阵，数目由framelen决定
    # winlen代表窗口的时间跨度，winstep表示每次窗口前进的时间
    # frames return Returns: an array of frames. Size is NUMFRAMES by frame_len.(这个函数利用窗函数对信号进行了分割)

    spec_list = list()
    for sig in sigs:
        frames = sigproc.delay_ica_framesig(sig, frame_len=winlen * rate, frame_step=winstep * rate, winfunc=winfuc[fuc_name])

        mag = sigproc.magspec(frames, NFFT)
        # 习惯上我们将频谱值表现在y轴上，故旋转
        mag = np.rot90(mag)
        spec_list.append(mag)

    spec_list = np.asarray(spec_list)
    return spec_list

# 以下参数都直接来源于kahst's code
# NFFT 通常取2的幂次
def GetMultiMag(path, ICA, second=5, overlap=2, minlen=1, winsecond=3, winlen=0.05, winstep=0.0097, NFFT=1024, rate=44100):
    # 考虑物种只有10种，数据量不大，减少second生成更多spec

    '''
        :param path: the audio data path
        :param second: we will cut the audio into chunks whose duration is param 'second'
        :param WinlenList: it used for multi scale, it's a list with time data.
        :param NFFT: the num of FFT
        :return: return cepstrum list including three pic though some of them will be zeros cause there is function to check if there are enough valid signals in one spec
        :param overlap: it should be a param but I delete it cause somewhere make it as half of second. And I ues it too
        Attention: cause we get multi-scale spectrum so for the same output size.I add one more step to get cepstrum.
        '''
    # Read the audio
    sample_rate, sig = wave.read(path)
    # TODO: to get the speices as the file_name
    # name = path
    sig = ChangeSample(sig, rate=rate)

    # split into chunks with overlap
    sig_splits = []
    # overlap = second*0.5
    for i in range(0, len(sig), int((second - overlap) * rate)):
        split = sig[i:i + second * rate]
        if len(split) < minlen*rate:
            pass
        elif len(split) < second * rate:
            # We will quit the short split before, but this time we save it with appending zero points.
            z = np.zeros(second * rate - len(split))
            split = np.append(split, z)
        sig_splits.append(split)


    # Doing fourier transform
    if len(sig_splits) == 0:
        return [0], [0]
    else:
        # TODO:there is a question is about how to make the length in temporal axis is same to others?
        Spec_list = []

        # multiprocess function

        for buff in sig_splits:
            # 在获取频谱图之前我们需要进行预加重
            pre = psf.sigproc.preemphasis(buff, coeff=0.95)

            # ICA 提取成分
            pre = single_to_multi(pre, target_channel=5)
            component = ICA.fit_transform(pre)
            # TODO: The shape of pre ? hope to be length*n_component
            pre = np.dot(component.T, pre).T # return length*n_component

            # 将buff转换为相应的频谱图
            MagSpec = GetMag(pre, RATE, winlen, winstep, NFFT, fuc_name='Hanning')
            # Get the power spec
            MagSpec = abs(np.multiply(MagSpec, MagSpec))
            # Normalize Process
            MagSpec -= MagSpec.min(axis=None)
            MagSpec /= MagSpec.max(axis=None)
            # MagSpec = MagSpec[:256, :512]
            magspec = cv2.resize(MagSpec, (512, 256))

            # plt.imshow(MagSpec)
            # plt.show()
            # MagSpec *= 255
            has_bird = hasbird(magspec)
            # log_Mag = np.log(MagSpec+1e-10)
            # log_Mag = log_Mag.astype('float32')
            # CepSpec = cv2.dct(log_Mag)
            # we are prior to choose the low frequency data
            # TODO:we can do some IDCT and calculate the simularity to prove the number choosed is right.
            Spec_list.append([has_bird, buff, magspec])

        return Spec_list


if __name__ == '__main__':
    # hyperparameter setting
    WinLen = 0.05
    WinStep = 0.0097
    noise = 0

    n_component = 5
    iters = 500

    # species limit
    limit = ['devillei', 'leucostigma','fuscicauda','obsoletum','striaticollis','serva','leucophrys','mentalis','sulphuratus','ferruginea']

    df_path = r'G:\dataset\BirdClef\vacation\static.csv'
    src_dir = r'G:\dataset\BirdClef\paper_dataset\wav'
    spec_dir = r'G:\dataset\BirdClef\vacation\spec'
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir)

    df = pd.read_csv(df_path)
    df = df[df.Species.isin(limit)]
    species_list = list(set(df.Species))
    print('{} the trasform species is {}'.format(datetime.now(), species_list))

    noise_index = 0
    print('Preprocessing start')
    ICA_transofmer = FastICA(n_component=n_component, max_iter=iters)
    for item in df.iterrows():
        # Get the wav path
        item = item[1]
        FileName = item['FileName']
        file_path = os.path.join(src_dir, FileName)
        # Get the audio save path
        wav_name = FileName.split('.')[0]

        wav_info = GetMultiMag(file_path, ICA_transofmer)
        for index, info in enumerate(wav_info):
            has_bird, buff, img = info[0], info[1], info[2]
            # every spec_list
            if has_bird:
                spec_name = wav_name + '-{}'.format(index) + '.png'
                cv2.imwrite(os.path.join(spec_dir, spec_name), img * 255)
            else:
                noise_name_img = wav_name + 'noise-{}'.format(noise_index) + '.png'
                noise_index += 1
                cv2.imwrite(os.path.join(spec_dir, noise_name_img), img * 255)
    print('Done.')