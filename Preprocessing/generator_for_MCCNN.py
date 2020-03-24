import pandas as pd
import os
import matplotlib.pyplot as plt
import random

import numpy as np
import cv2

import threading
import time

current_path = os.getcwd()

class generator():
    def __init__(self, Img_size, label_path):
        '''
        :param label_path: the file including kinds' info (Family:['af','bf'...],Genus:['ag','bg',...])
        '''

        # label_info is a dict that keys is the name of (family\genus\species) and the values is digit ID
        self.data_info = pd.read_csv(label_path)
        self.label_info = self._label_info(label_path)

        self.RANDOM = np.random.RandomState(3)
        self.IM_AUGMENTATION = {  # 'type':[probability, value]
            #'roll': [0.3, (0.0, 0.05)],
            'noise': [0.2, 0.01],
            # 'noise_samples': [0.2, 0.1],
            'brightness': [0.5, (0.25, 1.25)],
            'crop': [0.4, 0.07],
            'flip': [0.3, 1]
        }
        self.IM_SIZE = Img_size # W, H

    def __call__(self, *args, **kwargs):
        return len(os.listdir(args[0])), self.data_gener(*args, **kwargs)

    def data_gener(self, spec_path, batchsize, aug):
        '''
        :param spec_path: From the data_file we get the FileName and this param get the full path of data
        '''

        data = os.listdir(spec_path)

        while True:
            i = 0
            random.shuffle(data)
            while i < len(data):
                # print('New round begining')
                filenames = data[i:i+batchsize]
                i += batchsize
                batchimg = []
                batchlabel = []
                # Get the the FileName's directory path
                for FileName in filenames:
                    filepath = os.path.join(spec_path, FileName)

                    if not os.path.exists(filepath):
                        # cause some audio will be regard as a noise audio
                        # And some species just get one or two audios
                        # So the file_dir will not exist
                        raise FileExistsError('{} dose not exist'.format(filepath))

                    img = self._img_get(filepath)
                    # print(img.shape)
                    if len(img.shape)==3 and img.shape != (self.IM_SIZE[0], self.IM_SIZE[1], self.IM_SIZE[2]):
                        print('The error file name is %s' % FileName)
                        # LIFECLEF2017_BIRD_XC_WAV_RN49356.wav
                        continue
                    elif len(img.shape)==2 and img.shape != (self.IM_SIZE[0], self.IM_SIZE[1],):
                        print('The error file name is %s' % FileName)
                        # LIFECLEF2017_BIRD_XC_WAV_RN49356.wav
                        continue

                    # self._imageAugmentation will do the augmentation job, but there is a probability that img doesn't get aug
                    if aug:
                        img = self._imageAugmentation(img, probability=0.8)

                    if len(img.shape) == 2:
                        img = np.expand_dims(img, 2)

                    # Save label
                    # The family/genus/species name
                    # todo: pure命名规则的问题
                    try:
                        label_name = FileName.split('.wav')[0]
                        label_name = self.data_info[self.data_info['FileName']==label_name+'.wav'].Species.values[0]
                    except:
                        label_name = FileName.split('-')[0]
                        label_name = self.data_info[self.data_info['FileName'] == label_name + '.wav'].Species.values[0]
                    label = [0] * len(self.label_info.keys())
                    label[int(self.label_info[label_name])] = 1

                    batchimg.append(img)
                    batchlabel.append(label)
                yield np.asarray(batchimg), np.asarray(batchlabel)

    def data_size(self, filep):
        file = pd.read_csv(filep)
        if self.limit:
            file = file[file.Species.isin(self.label_info.keys())]
        return file.shape[0]

    # The function get the img data from provided path
    def _img_get(self, filepath):
        #print('One Data Loading')
        img = plt.imread(filepath)
            # logging.error('The error file path is %s' %file_name)
            # print('The error file path is %s' % path)
        #print('One Data Loaded')
        img -= img.min()
        img /= img.max()
        return img*255

    def _return_noise(self, listdir=r'G:\dataset\BirdClef\paper_dataset\spectrum\noise'):
        RANDOM = self.RANDOM
        lists = os.listdir(listdir)
        file_path = RANDOM.choice(lists)
        return plt.imread(os.path.join(listdir, file_path))

    def _imageAugmentation(self, img, count=3, probability=0.7):
        RANDOM = self.RANDOM
        IM_SIZE = self.IM_SIZE
        AUG = self.IM_AUGMENTATION

        while (count >0 and len(AUG) > 0):
            if RANDOM.choice([True, False], p=[probability, 1 - probability]):
                # Random Crop (without padding)
                if'crop' in AUG and RANDOM.choice([True, False], p=[AUG['crop'][0], 1 - AUG['crop'][0]]):
                    h, w = img.shape[:2]
                    cropw = RANDOM.randint(1, int(float(w) * AUG['crop'][1]))
                    croph = RANDOM.randint(1, int(float(h) * AUG['crop'][1]))
                    img = img[croph:-croph, cropw:-cropw]
                    # TODO:注意cv2 resize顺序
                    img = cv2.resize(img, (IM_SIZE[1], IM_SIZE[0],))

                # Flip - 1 = Horizontal, 0 = Vertical
                elif 'flip' in AUG and RANDOM.choice([True, False], p=[AUG['flip'][0], 1 - AUG['flip'][0]]):
                    img = cv2.flip(img, AUG['flip'][1])

                # Wrap shift (roll up/down and left/right)
                elif 'roll' in AUG and RANDOM.choice([True, False], p=[AUG['roll'][0], 1 - AUG['roll'][0]]):
                    img = np.roll(img, int(img.shape[0] * (RANDOM.uniform(-AUG['roll'][1][1], AUG['roll'][1][1]))), axis=0)
                    img = np.roll(img, int(img.shape[1] * (RANDOM.uniform(-AUG['roll'][1][0], AUG['roll'][1][0]))), axis=1)

                # substrac/add mean
                elif 'mean' in AUG and RANDOM.choice([True, False], p=[AUG['mean'][0], 1 - AUG['mean'][0]]):
                    img += np.mean(img) * AUG['mean'][1]

                # gaussian noise
                # TODO:可以增加自己提取的噪声
                elif 'noise' in AUG and RANDOM.choice([True, False], p=[AUG['noise'][0], 1 - AUG['noise'][0]]):
                    img += RANDOM.normal(0.0, RANDOM.uniform(0, AUG['noise'][1] ** 0.5), img.shape)
                    img = np.clip(img, 0.0, 1.0)

                # add noise samples
                elif 'noise_samples' in AUG and RANDOM.choice([True, False], p=[AUG['noise_samples'][0], 1 - AUG['noise_samples'][0]]):
                    noise = self._return_noise()
                    img += noise*AUG['noise_samples'][1]
                    img -= img.min(axis=None)
                    img /= img.max(axis=None)

                # adjust brightness
                elif 'brightness' in AUG and RANDOM.choice([True, False], p=[AUG['brightness'][0], 1 - AUG['brightness'][0]]):
                    img *= RANDOM.uniform(AUG['brightness'][1][0], AUG['brightness'][1][1])
                    img = np.clip(img, 0.0, 1.0)
            count -= 1
        # show
        # cv2.imshow("AUG", img)#.reshape(IM_SIZE[1], IM_SIZE[0], IM_DIM))
        # cv2.waitKey(-1)

        return img

    def _label_info(self, label_path):
        try:
            df = pd.read_csv(label_path)
        except:
            df = pd.read_excel(label_path)
        label = set(df.Species.tolist())
        label_info = dict(zip(label, range(len(label))))
        return label_info


if __name__ == '__main__':
    label_path = r'G:\dataset\BirdClef\vacation\limit_species.csv'
    spec_path = r'G:\dataset\BirdClef\vacation\spectrum\limitspecs\train'
    source_train = generator(Img_size=[256, 256, 3], label_path=label_path)
    nums, source_train = source_train(spec_path, 8, True)
    next(source_train)


