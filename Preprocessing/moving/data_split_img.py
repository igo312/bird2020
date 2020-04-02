import os
import shutil
import random
from tqdm import tqdm
random.seed(3333)

if __name__ == '__main__':
    spec_dir = r'G:\dataset\BirdClef\vacation\spectrums_total\ICA30S1'
    train_dir = r'G:\dataset\BirdClef\vacation\spectrum\ICA30S1\train'
    validation_dir = r'G:\dataset\BirdClef\vacation\spectrum\ICA30S1\validation'
    test_dir = r'G:\dataset\BirdClef\vacation\spectrum\ICA30S1\test'

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    img_list = os.listdir(spec_dir)
    random.shuffle(img_list)
    # Split into three part: train, validation, test
    train = random.sample(img_list, round(len(img_list)*0.75))
    img_list = [i for i in img_list if i not in train]
    validation = random.sample(img_list, round(len(img_list)*0.7))
    test = [i for i in img_list if i not in validation]

    print('The length of train is {}, of validation is {}, of test is {}'.format(len(train),len(validation),len(test)))
    # Copy data into three parts
    for i in tqdm(train):
        src = os.path.join(spec_dir, i)
        dst = os.path.join(train_dir, i)
        shutil.copy(src, dst)

    for i in tqdm(validation):
        src = os.path.join(spec_dir, i)
        dst = os.path.join(validation_dir, i)
        shutil.copy(src, dst)

    for i in tqdm(test):
        src = os.path.join(spec_dir, i)
        dst = os.path.join(test_dir, i)
        shutil.copy(src, dst)

    print('Moving data Done')
