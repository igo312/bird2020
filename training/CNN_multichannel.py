# -*- encoding: utf-8 -*-

# TODO:原先的数据生成存在问题
import sys
sys.path.append(r'G:\bird2019')


from model.model_MCCNN import MCCNN, MOBV2
from model.mobilenetv3.mobilenet_v3_large import  MobileNetV3_Large
from Preprocessing.generator_for_MCCNN import generator
import argparse
import time
import os

import tensorflow as tf


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
    ap.add_argument('-l', '--lr', type=float, default=0.008, help="Initial Learning Rate")
    ap.add_argument('--batch_size', default=8, type=int, help='The input size that every iteration go through in.')
    ap.add_argument('--accumulation_steps', default=1, type=int, help='The input size that every iteration go through in.')
    ap.add_argument('--validation_step', default=324//8, type=int, help='The validation steps(val_num=step*batch) after a epoch')
    args = ap.parse_args()

    ######################################################################################################################
    # Get the data generator
    # class_num should be a num or None
    imgsize = [256,256,1]
    train= generator(imgsize, args.label_path)
    train_num, train = train(r'G:\dataset\BirdClef\vacation\spectrum\pure\train', args.batch_size, True)
    val = generator(imgsize, args.label_path)
    val_num, val= val(r'G:\dataset\BirdClef\vacation\spectrum\pure\validation', args.batch_size, False)
    test= generator(imgsize, args.label_path)
    test_num, test = test(r'G:\dataset\BirdClef\vacation\spectrum\pure\test', args.batch_size, False)

    #####################################################################################################################
    # Training procedure
    mobv2 = MOBV2(shape=imgsize, num_classes=10)
    #mobv3 = MobileNetV3_Large(imgsize, 10).build()
    # 模型起始速度比mobilenet快，是因为其参数少，误差就小些
    mccnn = MCCNN(args.lr, class_num=args.class_num, Img_size=imgsize, accumulation_steps=args.accumulation_steps)
    # model = mccnn.define_model()
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
        mccnn.train_model(train, val, mobv2,model_name='mobv2pure',
                                epochs=128, steps_per_epoch=train_num//args.batch_size,\
                                validation_steps=args.validation_step,\
                                start_epoch=0,
                                save_interval='epoch', #8*(train_num//args.batch_size//args.args.accumulation_steps)
                                save_path=args.save_path,
                                C_state=True,
                               )
        print('trainging Done, processing evaluating')
        mccnn.eval_source_classifier(test, mobv2)
    else:
        model = mccnn.get_source_classifier(mccnn.source_encoder, args.eval_source_classifier)
        mccnn.eval_source_classifier(test, model, 'source')
        #mccnn.eval_source_classifier(model, 'target')



