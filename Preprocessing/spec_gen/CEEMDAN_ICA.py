from PyEMD import CEEMDAN, Visualisation, EEMD, EMD
from sklearn.decomposition import FastICA
import numpy as np

import matplotlib.pyplot as plt
import time
import scipy.io.wavfile as wave
import random

class sigle_channel_ICA(object):
    # TODO ：分解的IMF数量可能少于n_component,( 添加原始信号用于ICA)，
    #  TODO：同时分解个数少于2则放弃？ 还要去掉一些没用的独立分量，学习论文
    def __init__(self, trails, n_component, whiten=True, max_iter=200):
        self.ceemdan = CEEMDAN(trials=trails)
        self.eemd = EEMD(trails=trails)
        self.emd = EMD()
        self.n_component = n_component
        self.ICA_transfomer = FastICA(n_components=n_component,
                                      whiten=whiten,
                                      max_iter=max_iter)

    def __call__(self, signal, method):
        if method == 'EMD':
            return self.EMD_ICA(signal)
        else:
            raise NameError('method must be "EMD", method "EEMD" or "CEEMADAN" will come in the future')
        '''
        if method == 'CEEMDAN':
            return self.CEEMDAN_ICA(signal)
        elif method == 'EEMD':
            return self.EEMD_ICA(signal)
        elif method == 'EMD':
            return self.EMD_ICA(signal)
        else:
            raise NameError('method must be "EMD", "EEMD" or "CEEMADAN"')
        '''

    def CEEMDAN_ICA(self, signal):
        cIMFs = self.ceemdan(signal)
        cIMFs = np.concatenate([cIMFs, np.expand_dims(signal, axis=0)])
        if cIMFs.shape[0] < self.n_component:
            print('the number of IMF is not enough as for ICA')
            return None
        component = self.ICA_transfomer.fit_transform(cIMFs)
        return np.dot(component.T, cIMFs)

    def EEMD_ICA(self, signal):

        eIMFs = self.eemd.eemd(signal)
        eIMFs = np.concatenate([eIMFs, np.expand_dims(signal,axis=0)])
        if eIMFs.shape[0] < self.n_component:
            print('the number of IMF is not enough as for ICA')
            return None
        component = self.ICA_transfomer.fit_transform(eIMFs)
        return np.dot(component.T, eIMFs)

    def EMD_ICA(self, signal):
        #import time
        #a = time.time()

        #imfs, res = self.emd.get_imfs_and_residue()
        #from PyEMD import EMD, Visualisation
        #t = np.arange(0,0.05,1/22050)
        #vis = Visualisation()
        #vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
        #vis.plot_instant_freq(t, imfs=imfs)
        #vis.show()

        IMFs = self.emd.emd(signal)
        IMFs = self._frequency_check(IMFs)
        # IMFs = np.concatenate([IMFs[:-1], np.expand_dims(signal,axis=0)])
        if IMFs.shape[0] < self.n_component or len(IMFs.shape)==1:
            print(r'the number of IMF is not enough as for ICA')
            IMFs = self._ICA_expand(IMFs)
        #c = time.time()
        componet = self.ICA_transfomer.fit_transform(IMFs)
        #b = time.time()
        #print('ICA transform Done, EMD consume {:.2f} seconds, ICA consume {:.2f} seconds'.format(c-a, b-c))
        return np.dot(componet.T, IMFs).T

    def _frequency_check(self, imfs):
        new_imfs = list()
        for imf in imfs:
            imf_fre = np.fft.rfft(imf)
            # 100*20 = 2000Hz
            # np.sum(imf_fre[:100]) / np.sum(imf_fre)
            low_fre_percentage = abs(np.sum(imf_fre[:25])) / (abs(np.sum(imf_fre))+0.01)
            if low_fre_percentage < 0.7:
                new_imfs.append(imf)
        if not new_imfs:
            return  imfs[0]
        return np.asarray(new_imfs)

    def _ICA_expand(self, imfs):
        if len(imfs.shape) == 1:
            imfs = np.expand_dims(imfs, axis=0)
        while imfs.shape[0] < self.n_component:
            i = random.randint(0, imfs.shape[0]-1)
            temp = imfs[i]
            cut_index = random.randint(0, len(temp)-1)
            imfs = np.concatenate(
                [imfs,np.expand_dims(np.append(temp[cut_index:], temp[:cut_index]),axis=0)],
                axis=0
            )
        return imfs

if __name__ == '__main__':
    path1 = r'F:\鸟鸣\背景大于鸟声LIFECLEF2014_BIRDAMAZON_XC_WAV_RN1245.wav'
    path2 = r'F:\鸟鸣\多种物种，目标时间短LIFECLEF2014_BIRDAMAZON_XC_WAV_RN757.wav'
    path3 = r'F:\鸟鸣\较清楚音频LIFECLEF2014_BIRDAMAZON_XC_WAV_RN444.wav'

    # 通过观察时域波形和播放sig1，可以发现背景音的量可能大于鸟鸣，时域上无法良好体现，但在时域上突然出现变化是鸟鸣的出现，但有所遗漏
    rate, sig1 = wave.read(path1);
    # 波形展现了播放一样的结果，多种物种的声音频率相似，因此在时域上更加不明显的区分
    rate, sig2 = wave.read(path2);
    # 波形十分好，可以进行良好的识别
    rate, sig3 = wave.read(path3);

    sig1 = sig1.astype(np.int32)
    sig2 = sig2.astype(np.int32)
    sig3 = sig3.astype(np.int32)

    temp = sig1[:round(rate*1.5)]
    #ceemdan = CEEMDAN(trials=10)
    #eemd = EEMD(trials=1)
    #emd = EMD()


    #a = time.time()
    #emd = EMD()
    #IMFs = emd(temp)
    #b = time.time()
    #print('EMD consumes time is %.4f' % (b - a))

    a = time.time()
    ceemdan = CEEMDAN(trials=10)
    cIMFs = ceemdan(temp)
    b = time.time()
    print('CEEMDAN consumes time is %.4f'%(b-a))

    transformer = FastICA(n_components=3,
                         random_state = 0,
                          whiten=True,
                         max_iter=1000)
    transform_component = transformer.fit_transform(cIMFs, )

    sperated_signale = np.dot(transform_component.T, cIMFs)
    plt.subplot(4,1,1)
    plt.plot(sig1)

    for index, sig_temp in enumerate(sperated_signale):
        plt.subplot(4,1,index+2)
        plt.plot(sig_temp)

    a = time.time()
    eemd = EEMD(trials=5)
    eIMFs = eemd(temp)
    b = time.time()
    print('EEMD consumes time is %.4f' %(b - a))

# 100次 CEEMDAN 29s EEMD 14s
# 20次 CEEMDAN 7s   EEMD 3s
# #TODO： 太慢了，要用到多线程了

    vis = Visualisation()
