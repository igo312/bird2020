import pandas as pd
import os
import matplotlib.pyplot as plt
import random

import numpy as np
import cv2

from sklearn.utils import shuffle
import threading
import time

current_path = os.getcwd()

class Data_Gener():
    def __init__(self, Img_size, label_path, limit_species=None):
        '''
        :param label_path: the file including kinds' info (Family:['af','bf'...],Genus:['ag','bg',...])
        '''

        # label_info is a dict that keys is the name of (family\genus\species) and the values is digit ID
        self.label_info = self._label_info(label_path)

        # To limit the species in case to check whether the model get the optimal
        self.limit = limit_species
        if limit_species:
            self.label_info = dict(zip(
                list(self.label_info.keys())[:limit_species],
                range(limit_species)
            ))

        self.RANDOM = np.random.RandomState(3)
        self.IM_AUGMENTATION = {  # 'type':[probability, value]
            #'roll': [0.3, (0.0, 0.05)],
            'noise': [0.2, 0.01],
            'noise_samples': [0.2, 0.1],
            'brightness': [0.5, (0.25, 1.25)],
            'crop': [0.4, 0.07],
            'flip': [0.3, 1]
        }
        self.IM_SIZE = Img_size # W, H

    def data_gener(self, spec_path, batchsize, aug):
        '''
        :param spec_path: From the data_file we get the FileName and this param get the full path of data
        '''

        img_list = os.listdir(spec_path)

        i = 0
        data = shuffle(img_list)
        while i < len(data):
            # print('New round begining')
            filenames = data[i:i+batchsize]
            i += batchsize
            batchimg = np.array()
            batchlabel = np.array()
            # Get the the FileName's directory path
            for FileName in filenames:
                filepath = os.path.join(spec_path, FileName.split('.')[0])

                if not os.path.exists(filepath):
                    # cause some audio will be regard as a noise audio
                    # And some species just get one or two audios
                    # So the file_dir will not exist
                    raise FileExistsError('{} dose not exist'.format(filepath))

                img = self._img_get(filepath)
                # print(img.shape)
                if img.shape != (self.IM_SIZE[0], self.IM_SIZE[1]):
                    print('The error file name is %s' % FileName)
                    # LIFECLEF2017_BIRD_XC_WAV_RN49356.wav
                    continue

                # self._imageAugmentation will do the augmentation job, but there is a probability that img doesn't get aug
                if aug:
                    img = self._imageAugmentation(img, probability=0.8)

                # Save label
                # The family/genus/species name
                # todo: 查看label对不对的上
                label_name = FileName.split('.wav')[0]
                label = [0] * len(self.label_info.keys())
                label[int(self.label_info[label_name])] = 1

                if not np.any(batchimg):
                    batchimg = img
                    batchlabel = label
                else:
                    batchimg = np.append(batchimg,img)
                    batchlabel = np.append(batchlabel, label)
            yield batchimg, batchlabel

    def data_size(self, filep):
        file = pd.read_csv(filep)
        if self.limit:
            file = file[file.Species.isin(self.label_info.keys())]
        return file.shape[0]

    # The function get the img data from provided path
    def _img_get(self, filepath):
        #print('One Data Loading')
        img = plt.imread(filepath)
        IM_SIZE = self.IM_SIZE
        if img.shape != (IM_SIZE[0], IM_SIZE[1]):
            # TODO: It may be caused because of different win_len and I forget to resize it
            # The error img shape is (400, 109)
            # The error img shape is (400, 235)
            # print('The error img shape is {}'.format(img.shape))
            print('Occurs error img {}'.format(filepath))
            img = cv2.resize(img, (IM_SIZE[1], IM_SIZE[0]))

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

    # In one training time, This function will be called only one time
    def _label_info(self, label_path):
        try:
            df = pd.read_csv(label_path)
        except:
            df = pd.read_excel(label_path)
        label = set(df.Species.tolist())
        label_info = dict(zip(label, range(len(label))))
        return label_info

        # Label genor, using one time is enough, return csv file that includes three kinds of label


if __name__ == '__main__':
    label_path = r'G:\dataset\BirdClef\vacation\limit_species.csv'
    spec_path = r'G:\dataset\BirdClef\vacation\mat'
    source_train = Data_Gener(Img_size=[256, 256], label_path=label_path, limit_species=10)
    source_train = source_train.data_gener(spec_path, 8, True)

    next(train_source)
    #train_gen1, train_num1 = gener.data_gener(data_file_path=data_file_path1, BatchSize=8, label_type='source',
    #                                      spec_path=spec_path, mode=mode, aug=False, use_thread=True)
    #train_gen2, train_num2 = gener.data_gener(data_file_path=data_file_path2, BatchSize=8, label_type='target',
    #                                          spec_path=spec_path, mode=mode, aug=False, use_thread=False)
    num = 0
    start1 = time.time()
    while num < 300:
        #if num % 10000000 == 0:
        #    print('The next num is %d'%num)
        _, _ = next(test_source)
        #print('The label {} \n'.format(np.argmax(labels, axis=1)))
        num += 1
        #print(labels.argmax(axis=1))
    end1 = time.time()
    consume1 = end1 - start1

    start2 = time.time()
    while num < 200:
        #if num % 10000000 == 0:
        #    print('The next num is %d'%num)
        _, _ = next(train_source)
        #print('The label {} \n'.format(np.argmax(labels, axis=1)))
        num += 1
        #print(labels.argmax(axis=1))
    end2 = time.time()
    consume2 = end2 - start2
    print('The time that not use safe-thread consume {}'.format(consume1))
    print('The time that use safe-thread consume {}'.format(consume2))

