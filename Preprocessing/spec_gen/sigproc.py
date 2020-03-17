# This file includes routines for basic signal processing including framing and computing power spectra.
# Author: James Lyons 2012
import decimal

import numpy
import math
import logging
import random

from multiprocessing import Pool


def round_half_up(number):
    # 四舍五入
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def delay_ica_framesig(sig,frame_len,frame_step, winfunc=lambda x:numpy.ones((x,))):
    slen = sig.shape[0]
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    # 为了保证最后一个窗口到长度，使用pad补充
    padlen = int((numframes-1) * frame_step + frame_len)
    zeros = numpy.zeros((padlen - slen,sig.shape[1]))
    padsignal = numpy.concatenate((sig, zeros))

    # 为了防止最后一个pad有大量的0导致的IMF失败
    # numframes -= 1

    # 一行为一个窗口长度，并且得到正确的索引 [[frame_step*0, frame_step*0+1,...,frame_step*0+frame_len],
    #                                     [frame_step*1, frame_step*1+1,...,frame_step*1+frame_len],...,]
    # TODO:得调试这一块，正确复制数据
    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1, 3)) + \
              numpy.tile(numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1, 3)).T
    indices = numpy.array(indices, dtype=numpy.int32)

    # EMD和ICA操作,使用多进程
    frames = padsignal[indices]


    # 加窗
    win = numpy.tile(winfunc(frame_len), (numframes, 1))
    frames = []
    for i in range(frames.shape[-1]):
        frames.append(frames[:, :, i] * win)
    frames = numpy.asarray(frames)
    frames = numpy.transpose(frames, (1, 2, 0))
    return frames

def emd_ica_framesig(sig,frame_len,frame_step, ICA_transformer, winfunc=lambda x:numpy.ones((x,))):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    # frame_len: 窗口长度点数
    # frame_step: 窗口移动点数
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))

    # 为了保证最后一个窗口到长度，使用pad补充
    padlen = int((numframes-1)*frame_step + frame_len)
    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig,zeros))

    # 为了防止最后一个pad有大量的0导致的IMF失败
    numframes -= 1

    # 一行为一个窗口长度，并且得到正确的索引 [[frame_step*0, frame_step*0+1,...,frame_step*0+frame_len],
    #                                     [frame_step*1, frame_step*1+1,...,frame_step*1+frame_len],...,]
    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) +\
              numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)

    # EMD和ICA操作,使用多进程
    frames = padsignal[indices]

    if ICA_transformer:
        ICA_frames = []
        import time
        a = time.time()
        print('ICA transform beging')
        try:
            for frame in frames:
                ICA_frames.append(ICA_transformer(frame, method='EMD'))
        except:
            print('ICA transofm break down')
            return None
        b = time.time()
        print('ICA transform Done, consume %.2f seconds'%(b-a))
        ICA_frames = numpy.asarray(ICA_frames)
        #pool = Pool(2)
        #pool.map(ICA_transform, frames)
        #pool.close()
        #pool.join()

    # 加窗
    win = numpy.tile(winfunc(frame_len),(numframes,1))
    frames = []
    for i in range(ICA_frames.shape[-1]):
        frames.append(ICA_frames[:,:,i]*win)
    frames = numpy.asarray(frames)
    frames = numpy.transpose(frames, (1,2,0))
    return frames


def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:numpy.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    padlen = (numframes-1)*frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]

    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]

def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warning('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.', numpy.shape(frames)[1], NFFT)
    specs = []
    for i in range(frames.shape[-1]):
        specs.append(numpy.rot90(
            numpy.absolute(
            numpy.fft.rfft(frames[:,:,i],NFFT)
        )))
    specs = numpy.transpose(numpy.asarray(specs), (1,2,0))
    return specs

def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0/NFFT * numpy.square(magspec(frames,NFFT))

def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps

def preemphasis(signal,coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0],signal[1:]-coeff*signal[:-1])



