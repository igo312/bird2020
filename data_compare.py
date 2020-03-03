# -*- encoding:utf-8 -*-
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
from scipy import interpolate, ndimage
import numpy as np
import python_speech_features as psf
import cv2
import os
from PIL import ImageEnhance
from PIL import Image

RATE = 44100
def plot_time(sigs1, sigs2, sigs3):
    sig1, stop1 = sigs1
    sig2, stop2 = sigs2
    sig3, stop3 = sigs3

    plt.subplot(311)
    plt.plot(np.linspace(0, stop1, len(sig1)), sig1)
    plt.title("The sound value of background is bigger than bird's .wav")

    plt.subplot(312)
    plt.plot(np.linspace(0, stop2, len(sig2)), sig2)
    plt.title("A lot of species, and the time of bird is short.wav")

    plt.subplot(313)
    plt.plot(np.linspace(0, stop3, len(sig3)), sig3)
    plt.title('A good record.wav')

    plt.subplots_adjust(hspace=0.8)

def ChangeSample(sig, rate):
    if rate == RATE:
        return sig

    duration = sig / rate
    x_old = np.linspace(0, duration, duration*rate)
    x_new = np.linspace(0, duration, duration*RATE)

    inter = interpolate.interp1d(x_old, sig)
    new_sig = inter(x_new)
    # TODO：原先使用round是不是造成了太大的误差，decimal没有设置
    new_sig = np.round(new_sig, decimals=1)
    return new_sig

def GetMag(sig, rate, winlen, winstep, NFFT, fuc_name='Rect'):
    winfuc = {
        'Rect': lambda x: np.ones((x,)),
        'Hamming': lambda x: np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (x - 1)) for n in range(x)]),
        'Hanning': lambda x: np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (x - 1)) for n in range(x)])
    }
    frames = psf.sigproc.framesig(sig, frame_len=winlen * rate, frame_step=winstep * rate, winfunc=winfuc[fuc_name])
    mag = psf.sigproc.magspec(frames, NFFT)
    mag_ = mag.copy()
    mag_ = norm(mag)
    plt.imshow(mag,)
    # 习惯上我们将频谱值表现在y轴上，故旋转
    # todo: 为什么计算过来好像原点就是放在最上面的？
    a = np.rot90(mag)
    plt.subplot(211)
    plt.imshow(mag)
    plt.subplot(212)
    plt.imshow(a)
    # a = np.fliplr(a)
    #TODO 感觉翻转还是错的...
    return a

def GetMultiMage(save_path, path, second=5, overlap=1, WinLen=0.04, WinStep=0.0097, NFFT=2048, minlen=3):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rate, sig = wave.read(path)
    # TODO: to get the speices as the file_name
    # name = path
    sig = ChangeSample(sig, rate=rate)
    # split into chunks with overlap
    sig_splits = []
    # overlap = second*0.5
    for i in range(0, len(sig), int((second - overlap) * rate)):
        # 分割之后将出现某些片段过短，这是不需要的故需要加入判断句
        split = sig[i:i + second * rate]
        if len(split) != second * rate:
            # We will quit the short split before, but this time we save it with appending zero points.
            z = np.zeros(second*rate - len(split))
            temp = np.append(split, z)
            sig_splits.append(temp)
        else:
            sig_splits.append(split)

    if len(sig_splits) == 0:
        return [0], [0]
    else:
        # TODO:there is a question is about how to make the length in temporal axis is same to others?
        Spec_list = []
        for index, buff in enumerate(sig_splits):
            # 在获取频谱图之前我们需要进行预加重
            pre = psf.sigproc.preemphasis(buff, coeff=0.95)
            # 将buff转换为相应的频谱图
            MagSpec = GetMag(pre, RATE, WinLen, WinStep, NFFT, fuc_name='Hanning')
            # Get the power spec
            #TODO: 功率谱有点不合适， 让信息不明显

            # MagSpec = abs(np.multiply(MagSpec, MagSpec))
            # Normalize Process
            #MagSpec -= MagSpec.min(axis=None)
            #MagSpec /= MagSpec.max(axis=None)
            # TODO 信息丢失了
            #MagSpec = MagSpec[:256, :512]
            #temp = np.zeros((256, 512), dtype="float32")
            #temp[:MagSpec.shape[0], :MagSpec.shape[1]] = MagSpec
            #magspec = temp.copy()
            # cv2.imwrite 矩阵大小操作和 np.shape相反
            magspec = cv2.resize(MagSpec, (256, 512))
            magspec -= magspec.min(axis=None)
            magspec /= magspec.max(axis=None)
            # magspec1 = MagSpec.resize((512, 256))
            #Spec_list.append(magspec)
            # cv2.imwrite must use path in english
            plt.imsave(os.path.join(save_path, str(index)+'.png'), magspec)
            #plt.imsave(os.path.join(save_path, str(index) + '1.png'), magspec*255)
            #cv2.imwrite(os.path.join(save_path, str(index)+'.png'), magspec*255)
            # TODO：cv2.需要乘255， 需要使用英文路径才能保存， plt两者都不需要
            #cv2.imwrite(os.path.join('F:\\', str(index) + '1.png'), magspec*255)
            #cv2.imwrite(os.path.join('F:\\', str(index) + '.png'), magspec)
        return

def norm(spec):
    spec -= spec.min()
    spec /= spec.max()
    return spec

if __name__ == '__main__':
    path1 = r'F:\鸟鸣\背景大于鸟声LIFECLEF2014_BIRDAMAZON_XC_WAV_RN1245.wav'
    path2 = r'F:\鸟鸣\多种物种，目标时间短LIFECLEF2014_BIRDAMAZON_XC_WAV_RN757.wav'
    path3 = r'F:\鸟鸣\较清楚音频LIFECLEF2014_BIRDAMAZON_XC_WAV_RN444.wav'

    # 通过观察时域波形和播放sig1，可以发现背景音的量可能大于鸟鸣，时域上无法良好体现，但在时域上突然出现变化是鸟鸣的出现，但有所遗漏
    rate, sig1 = wave.read(path1); stop1 = sig1.shape[0] / rate

    # 波形展现了播放一样的结果，多种物种的声音频率相似，因此在时域上更加不明显的区分
    rate, sig2 = wave.read(path2); stop2 = sig2.shape[0] / rate

    # 波形十分好，可以进行良好的识别
    rate, sig3 = wave.read(path3); stop3 = sig3.shape[0] / rate

    # GetMultiMage(r'F:\鸟鸣\多种物种，目标时间短', path2)

    #a = np.load(r'f:\\鸟鸣\with_blank1.npy')
    #spec = norm(GetMag(a, rate=44100, winlen=0.02, winstep=0.01, NFFT=1024, fuc_name='Hanning'))

    spec1 = norm(GetMag(sig1, rate=44100, winlen=0.02, winstep=0.01, NFFT=1024))
    spec2 = norm(GetMag(sig2, rate=44100, winlen=0.02, winstep=0.01, NFFT=1024))
    spec3 = norm(GetMag(sig3, rate=44100, winlen=0.02, winstep=0.01, NFFT=1024))

    plt.imsave(r'f:\\鸟鸣\背景大于鸟声.png', (1-spec1)*255,cmap='gray')
    plt.imsave(r'f:\\鸟鸣\多种物种.png', 1-spec2,cmap='gray')
    plt.imsave(r'f:\\鸟鸣\较好音频.png', 1-spec3,cmap='gray')

    s1 = plt.imread(r'f:\\鸟鸣\背景大于鸟声.png')
    plt.subplot(211)
    plt.imshow(spec1)
    plt.subplot(212)
    plt.imshow(1-spec1, cmap='gray')
    plt.imshow(s1)

    '''
    TODO:存在 pic has a wrong mode 问题，使用pic.mode可以确定其mode
    pil1 = (spec1*255).astype(np.int32)
    pil1 = Image.fromarray(spec1)
    enhance1 = ImageEnhance.Contrast(pil1)
    en1 = enhance1.enhance(1.5)
    '''


    # pre = psf.sigproc.preemphasis(sig1, coeff=0.95)
    GetMultiMage(r'F:\鸟鸣\背景大于鸟声',path1)
    GetMultiMage(r'F:\鸟鸣\多种物种，目标时间短', path2)
    GetMultiMage(r'F:\鸟鸣\较清楚音频', path3)

'''
plt.subplot(131)
plt.imshow(MagSpec, origin='lower')

plt.subplot(132)
plt.imshow(a, origin='lower')

plt.subplot(133)
plt.imshow(magspec, origin='lower')

plt.xticks(range(a.shape[1]),[1,2,3])

plt.subplot(211)
plt.plot(sig2)
plt.yticks([])
plt.xticks(np.linspace(0, len(sig2), 7), np.linspace(0, len(sig2)/44100, 7))
plt.xticks([])
plt.subplot(212)
plt.imshow(spec2, cmap=plt.get_cmap('gray_r'))
plt.yticks([])
plt.xticks(np.linspace(0, len(sig2), 7), np.linspace(0, len(sig2)/44100, 7))
'''
