from scipy.io import loadmat
import python_speech_features as psf
import numpy as np
import matplotlib.pyplot as plt
import librosa

rate = 44100
winlen = 0.05
winstep = 0.0097
NFFT = 1024

path = r'G:\dataset\BirdClef\vacation\ica\LIFECLEF2014_BIRDAMAZON_XC_WAV_RN6.wav'
mat = loadmat(path)
data = mat['component']

mags = []
for d in range(3):
    mag = librosa.core.stft(np.asfortranarray(data[d]), n_fft=512, hop_length=128, win_length=512,window='hann')
    mag = abs(mag)
    mag = mag - mag.min()
    mag = mag / mag.max()
    mags.append(mag)

mags = np.concatenate([
    np.expand_dims(mags[0],axis=2),
    np.expand_dims(mags[1],axis=2),
    np.expand_dims(mags[2],axis=2),
], axis=2)

i = 2200*4
plt.imshow(mags[:,i:i+1024])
# 习惯上我们将频谱值表现在y轴上，故旋转
print('matrix size is',mag.shape)

plt.imshow(mag)


path1 = r'f:\python.png'
path2 = r'f:\matlab.png'
img1 = plt.imread(path1)
img2 = plt.imread(path2)