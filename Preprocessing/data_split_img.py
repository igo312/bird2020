import os
import shutil
import random

if __name__ == '__main__':
    spec_dir = r'G:\dataset\BirdClef\vacation\mat'
    train_dir = r'G:\dataset\BirdClef\vacation\spectrum\limitspecs\train'
    validation_dir = r'G:\dataset\BirdClef\vacation\spectrum\limitspecs\validation'
    test_dir = r'G:\dataset\BirdClef\vacation\spectrum\limitspecs\test'

    img_list = os.listdir(spec_dir)
    random.shuffle(img_list)
    # Split into three part: train, validation, test
    train = random.sample(img_list, round(len(img_list)*0.7))
    img_list = [i for i in img_list if i not in train]
    validation = random.sample(img_list, round(len(img_list)*0.7))
    test = [i for i in img_list if i not in validation]

    print('The length of train is {}, of validation is {}, of test is {}'.format(len(train),len(validation),len(test)))
    # Copy data into three parts
    for i in train:
        src = os.path.join(spec_dir, i)
        dst = os.path.join(train_dir, i)
        shutil.copy(src, dst)

    for i in validation:
        src = os.path.join(spec_dir, i)
        dst = os.path.join(validation_dir, i)
        shutil.copy(src, dst)

    for i in test:
        src = os.path.join(spec_dir, i)
        dst = os.path.join(test_dir, i)
        shutil.copy(src, dst)

    print('Moving data Done')
