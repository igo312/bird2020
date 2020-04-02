# -*- encoding: utf-8 -*-

# TODO:原先的数据生成存在问题
import sys
sys.path.append(r'G:\bird2019')

from keras.model.model_MCCNN import MCCNN
from keras.model.mobilenetv2 import MOBV2
from Preprocessing.generator.generator_for_MCCNN import generator

from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)

import argparse

if __name__ == '__main__':

    # 命令行该怎么输入？:
    # python main_mccnn.py -参数1缩写 参数1输入 -参数2缩写 参数2输入

    ap = argparse.ArgumentParser()
    # 前面的 ‘-s’ 格式是缩写指代参数名
    ap.add_argument('-s', '--source_weights', required=False,
                    help="Path to weights file to load source model for training classification/adaptation")
    ap.add_argument('-e', '--start_epoch', type=int, default=1, required=False,
                    help="Epoch to begin training source model from")
    ap.add_argument('-n', '--discriminator_epochs', type=int, default=10000,
                    help="Max number of steps to train discriminator")
    # TODO:这里应该是在命令行中进行控制，应该选个方法是在idle中进行控制
    ap.add_argument('-a', '--source_discriminator_weights', help="Path to weights file to load source discriminator")
    ap.add_argument('-b', '--target_discriminator_weights', help="Path to weights file to load target discriminator")
    ap.add_argument('-t', '--eval_source_classifier', default=None,
                    help="Path to source classifier model to test/evaluate")
    ap.add_argument('-d', '--eval_target_classifier', default=None,
                    help="Path to target discriminator model to test/evaluate")


    ap.add_argument('-clsN','--class_num', default=10,help='Decide the class num of model will yiled')
    ap.add_argument('-lps', '--label_path', default=r'G:\dataset\BirdClef\vacation\limit_species.csv')
    ap.add_argument('-SVP', '--save_path', default=r'G:\dataset\BirdClef\vacation\Checkpoint')
    ap.add_argument('-l', '--lr', type=float, default=0.005, help="Initial Learning Rate")
    ap.add_argument('--batch_size', default=8, type=int, help='The input size that every iteration go through in.')
    ap.add_argument('--accumulation_steps', default=1, type=int, help='The input size that every iteration go through in.')
    ap.add_argument('--validation_step', default=324//8, type=int, help='The validation steps(val_num=step*batch) after a epoch')
    args = ap.parse_args()
    train_path = r'G:\dataset\BirdClef\vacation\spectrum\ICAS1\train'
    val_path = r'G:\dataset\BirdClef\vacation\spectrum\ICAS1\val'
    test_path =  r'G:\dataset\BirdClef\vacation\spectrum\ICAS1\test'
    label_path = r'G:\dataset\BirdClef\vacation\limit_species.csv'

    imgsize = [256, 256, 3]

    ######################################################################################################################
    # Get the data generator
    # class_num should be a num or None

    train= generator(imgsize, args.label_path)
    train_num, train = train(r'G:\dataset\BirdClef\vacation\spectrum\ICAS1\train', args.batch_size, True)
    val = generator(imgsize, args.label_path)
    val_num, val= val(r'G:\dataset\BirdClef\vacation\spectrum\ICAS1\val'
                      r'idation', args.batch_size, False)
    test= generator(imgsize, args.label_path)
    test_num, test = test(r'G:\dataset\BirdClef\vacation\spectrum\ICAS1\test', args.batch_size, False)

    #####################################################################################################################
    # Training procedure

    #model = load_model(r'G:\dataset\BirdClef\vacation\Checkpoint\mobv1S1\mobv2S1-56-.hdf5')
    model = MOBV2(shape=imgsize, num_classes=10,weights=r'G:\dataset\BirdClef\vacation\Checkpoint\pickle-01-.hdf5')
    #model = MobileNetV3_Large(imgsize, 10).build()
    # model = mccnn.define_model()
    # 模型起始速度比mobilenet快，是因为其参数少，误差就小些
    mccnn = MCCNN(args.lr, class_num=args.class_num, Img_size=imgsize, accumulation_steps=args.accumulation_steps)
    args.eval_source_classifier = True
    if args.eval_source_classifier is None:
        if args.source_weights is None:
            print('training a new model.')
        print("training dataset's size is {}".format(train_num))
        # TODO:查看各个multiply是否正确
        #model = mccnn.define_model(args.source_weights)

        # model.summary() get the model information
        # we can enlarget the steps_per_epoch to see whether the model can converge
        # TODO:the tensorboard is different from version 1.0+, the 'val_acc' did not work
        # there is no saver schedular
        mccnn.train_model(train, val, model,model_name='pickle',
                                epochs=64, steps_per_epoch=train_num//args.batch_size//3.5,\
                                validation_steps=args.validation_step,\
                                start_epoch=0,
                                save_interval='epoch', #8*(train_num//args.batch_size//args.args.accumulation_steps)
                                save_path=args.save_path,
                                C_state=True,
                               )
        print('trainging Done, processing evaluating')
        mccnn.eval_source_classifier(test, model)
    else:
        print('Doing evaluation.')
        # model = mccnn.get_source_classifier(mccnn.source_encoder, args.eval_source_classifier)
        for _ in range(5):
            mccnn.eval_source_classifier(val, model)
        #mccnn.eval_source_classifier(model, 'target')



