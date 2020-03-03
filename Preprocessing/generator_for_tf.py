import pandas as pd
import os
import matplotlib.pyplot as plt
import random

import numpy as np
import cv2

from sklearn.utils import shuffle
# from augmentation import augment
import threading
import time

# TODO: using  cv2.imread to get data
current_path = os.getcwd()
# logging.basicConfig(filename='my.log')
#TODO(11/13):在这里进行数据加强，在之前做数据做数据加强行不通

class Data_Gener():
    def __init__(self, mode, Img_size, label_path, limit_species=None):
        '''
        :param mode: To choose what kind the model base on ,(family\genus\species)
        :param label_path: the file including kinds' info (Family:['af','bf'...],Genus:['ag','bg',...])
        '''
        self.mode = mode

        # label_info is a dict that keys is the name of (family\genus\species) and the values is digit ID
        self.label_info = self._label_info(label_path)

        # To limit the species in case to check whether the model get the optimal
        self.limit = limit_species
        if limit_species:
            self.label_info  = dict(zip(
                list(self.label_info.keys())[:limit_species],
                range(limit_species)
            ))

        self.RANDOM = np.random.RandomState(3)
        self.IM_AUGMENTATION = {  # 'type':[probability, value]
            #'roll': [0.3, (0.0, 0.05)],
            'noise': [0.2, 0.01],
            'noise_samples': [0.4, 0.1],
            'brightness': [0.5, (0.25, 1.25)],
            'crop': [0.4, 0.07],
            # 'flip': [0.2, 1]
        }
        self.IM_SIZE = Img_size # W, H

    def data_gener(self, data_file_path, spec_path, aug):
        '''
        :param data_file_path: a path of pd.DataFrame instance including an item infos:
                                                        e.g. item:{'ClassID':fupfmb,
                                                                    'Family':Thamnophilidae
                                                                    'FileName':LIFECLEF2014_BIRDAMAZON_XC_WAV_RN100.wav
                                                                    'Genus':Hylophylax,
                                                                    'Species':punctulatus}
        :param spec_path: From the data_file we get the FileName and this param get the full path of data
        '''


        if type(data_file_path) == bytes:
            data_file_path = data_file_path.decode('ascii')
        file = pd.read_csv(data_file_path)

        if self.limit:
            file = file[file.Species.isin(self.label_info.keys())]
        data = file

        # cause tensorflow will post str in byte mode.
        if type(spec_path) == bytes:
            spec_path = spec_path.decode('ascii')
        data_path = spec_path

        # TODO: make a one image and label output file which delete `batch` element
        i = 0
        data = shuffle(data)
        data = data.reset_index(drop=True)
        while i < 425:
            # print('New round begining')
            item = data.iloc[i]
            i += 1

            # Get the the FileName's directory path
            FileName = item['FileName']
            # TODO:做一个文件夹的说明图(百度搜索目录结构图)
            file_dir = os.path.join(data_path, FileName.split('.')[0])

            # TODO(solved, Use data_check.py): Cause some file will not exist, the num of batch will be less than batch_size
            if not os.path.exists(file_dir):
                # cause some audio will be regard as a noise audio
                # And some species just get one or two audios
                # So the file_dir will not exist
                raise FileExistsError('{} dose not exist'.format(file_dir))

            img = self._img_get2(file_dir)
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
            label_name = item['Species']
            #
            label = [0] * len(self.label_info.keys())
            label[int(self.label_info[label_name])] = 1

            yield img, label


    # The function get the img data from provided path
    def _img_get2(self, FileDirPath):
        #print('One Data Loading')
        file_name = random.sample(os.listdir(FileDirPath), 1)[0]
        path = os.path.join(FileDirPath, file_name)
        img = plt.imread(path)
        IM_SIZE = self.IM_SIZE
        if img.shape != (IM_SIZE[0], IM_SIZE[1]):
            # TODO: It may be caused because of different win_len and I forget to resize it
            # The error img shape is (400, 109)
            # The error img shape is (400, 235)
            # print('The error img shape is {}'.format(img.shape))
            print('Occurs error img {}'.format(file_name))
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


                # substract/add mean
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
        '''
        def LabelCsv_genor(self, csv_file):
            # TODO: It's a rude way to genor label cause I am not familer with pandas and I want to save time
            df = pd.DataFrame()
            for kind in self.label_info:
                kind_data = list(set(list(csv_file[kind])))
                kind_dict = dict(zip(kind_data, range(len(kind_data))))
                df_temp = pd.DataFrame.from_dict(kind_dict, orient='index')
                temp = df_temp.reset_index()
                temp = temp.reset_index().rename(columns={'index': kind + 'Name', 0: kind + 'ID'})
                temp = temp.drop(['level_0'], axis=1)
                temp.to_csv(os.path.join(r'G:\dataset\BirdClef\paper_dataset', kind + '.csv'))
        '''

if __name__ == '__main__':
    label_path = r'G:\dataset\BirdClef\vacation\source.csv'
    mode = 'Species'
    data_file_path2 = r'G:\dataset\BirdClef\vacation\train_file\target'
    data_file_path1 = r'G:\dataset\BirdClef\vacation\train_file\source\source_train.csv'
    spec_path = r'G:\dataset\BirdClef\paper_dataset\spectrum'
    source_gener = Data_Gener(mode=mode, Img_size=[256, 512], label_path=label_path, limit_species=10)
    train_source = source_gener.data_gener(data_file_path=data_file_path1,
                                        spec_path=spec_path,aug=True)
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

