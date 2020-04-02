# -*- coding:utf-8 -*-
# 有论文采用了取倒谱的方法，在多尺度变换下，则将输入固定，不需要网络结构的调整，可以参考。但是倒谱对人不是很友好。
# 更改图片的保存路径，一种鸟类存放所有图片， 不再是一段音频。 图片命名规则 ‘音频原文件名-{:index}’，文件名用来索取label，index用于保存
# TODO：怎么消除一段音频的静音段？


'''
    这个程序将获取频谱图并进行一定程度的预处理
    ---------------------------------------
    获取频谱图：
    1.ChangeSample:检测音频的频率并保证其频率最终为44100Hz（44.1kHz）
    2.GetMag:将对应的输入转换经过傅立叶变换转换为相应的频谱图
    3.GetMutilMag：一段音频将多次采样获得多个频谱图，此函数即是对Changesample,getMag的集成运用，并遵守转换频谱图的流程（包含预加重）
    ---------------------------------------
    预处理：
    filter_isolated_cells:对于获取到的频谱图过滤独立的像素点
    hasbird:存在截取的音频纯为噪音因此需要判断此音频是否含有鸟声，主要依靠中位数以及ndimage.morphology.binary_dilation判断
    同时在此函数中集成filter_isolated_cells返回相应的判断状态，可命名为鸟声或者集成得到噪音
    cemdan_ICA：用于信号的分离后完成独立成分分析，然后用转换成频谱。
    -----------------------------------------
    集成：
    根据 http://blog.csdn.net/xmdxcsj/article/details/51228791
'''

from scipy import ndimage, interpolate
import python_speech_features as psf
import scipy.io.wavfile as wave
from librosa.core import stft

import cv2
import pandas as pd

from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm
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

def GetMag(sig, rate, winlen, winstep, NFFT, fuc_name='Rect'):
    '''获取输入音频的频谱图'''
    mag = stft(np.asfortranarray(sig), n_fft=NFFT, hop_length=int(winstep * rate), win_length=int(winlen * rate), window=fuc_name)

    # 习惯上我们将频谱值表现在y轴上，故旋转
    return mag


# 以下参数都直接来源于kahst's code
# NFFT 通常取2的幂次
def GetMultiMag(path, second=5, overlap=3, minlen=3, winlen=0.05, winstep=0.0097, NFFT=1024):
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
    rate, sig = wave.read(path)


    # sig = ChangeSample(sig, rate=rate)
    # 存在双声道信号
    if len(sig.shape) > 1:
        sigs = sig[:,0]
        for i in range(sig.shape[1]-1):
            sigs = np.concatenate([sigs, sig[:,i]], axis=0)
        sig = sigs

    # split into chunks with overlap
    sig_splits = []
    # overlap = second*0.5
    croplen = int(second * rate)
    for i in range(0, len(sig), int((second - overlap) * rate)):
        # 分割之后将出现某些片段过短，这是不需要的故需要加入判断句
        split = sig[i:i + second * rate]
        try:
            if len(split) != second * rate:
                # We will quit the short split before, but this time we save it with appending zero points.
                if len(split) < minlen * rate:
                    continue
                else:
                    zeros = np.zeros([croplen - split.shape[0]])
                    split = np.concatenate([split, zeros], axis=0)
        except:
            print('Filename is {}, shape is {}'.format(path.split('\\')[-1], sig.shape))
            raise ValueError
        sig_splits.append(split)

    Spec_list = []
    # multiprocess function
    for buff in sig_splits:
        # 在获取频谱图之前我们需要进行预加重
        pre = psf.sigproc.preemphasis(buff, coeff=0.95)

        # 将buff转换为相应的频谱图
        MagSpec = GetMag(pre, RATE, WinLen, WinStep, NFFT, fuc_name='hann')
        # Get the power spec
        MagSpec = abs(np.multiply(MagSpec, MagSpec))
        MagSpec = MagSpec[:512, :]
        # Normalize Process
        MagSpec -= MagSpec.min(axis=None)
        MagSpec /= MagSpec.max(axis=None)
        # MagSpec = MagSpec[:256, :512]
        magspec = cv2.resize(MagSpec, (256, 256))

        # plt.imshow(MagSpec)
        # plt.show()
        # MagSpec *= 255
        # has_bird = hasbird(magspec)
        # log_Mag = np.log(MagSpec+1e-10)
        # log_Mag = log_Mag.astype('float32')
        # CepSpec = cv2.dct(log_Mag)
        # we are prior to choose the low frequency data
        # TODO:we can do some IDCT and calculate the simularity to prove the number choosed is right.

        yield magspec

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
    noise = 0

    # species limit
    limit = ['devillei', 'leucostigma','fuscicauda','obsoletum','striaticollis','serva','leucophrys','mentalis','sulphuratus','ferruginea']

    df_path = r'G:\dataset\BirdClef\vacation\static.csv'
    src_dir = r'G:\dataset\BirdClef\paper_dataset\wav'
    spec_dir = r'G:\dataset\BirdClef\vacation\spectrums_total\pure'
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir)

    df = pd.read_csv(df_path)
    df = df[df.Species.isin(limit)]
    species_list = list(set(df.Species))
    # print('{} the trasform species is {}'.format(datetime.now(), species_list))

    strs = ''.join(os.listdir(spec_dir))

    file_index = 1
    print('Preprocessing start')

    for item in df.iterrows():
        # Get the wav path
        item = item[1]
        FileName = item['FileName']
        file_path = os.path.join(src_dir, FileName)
        # Get the audio save path
        wav_name = FileName.split('.')[0]

        if check(filename=wav_name, strs=strs):
            print('index-{} filename-{} already processed'.format(file_index, wav_name))
            file_index += 1
            continue

        wav_info = GetMultiMag(file_path, second=second, winlen=WinLen, winstep=WinStep, NFFT=NFFT)
        for index, img in enumerate(wav_info):
            has_bird = True
            # every spec_list
            if has_bird:
                spec_name = wav_name + '-{}'.format(index) + '.png'
                cv2.imwrite(os.path.join(spec_dir, spec_name), img * 255)
            #else:
            #    noise_name_img = wav_name + 'noise-{}'.format(noise_index) + '.png'
            #    noise_index += 1
            #    cv2.imwrite(os.path.join(noise_dir, noise_name_img), img * 255)
        print('第{}个文件处理完毕，文件名--{}'.format(file_index, FileName))
        file_index += 1
    print('Done.')