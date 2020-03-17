# -*- encoding:utf--8 -*-
# TODO: I think image should not be the good sample for ML method.So it will be fixed in feature, But I should try now

# -*- encoding: utf-8 -*-

# TODO:原先的数据生成存在问题
import sys
sys.path.append(r'G:\bird2019')


from Preprocessing.Training_Data_generator import Data_Gener
import argparse
from model.TCA import TCA
from model.MMD import mmd_linear
import time
import os

#from keras.backend import manual_variable_initialization
#manual_variable_initialization(True)

#save_path = r'G:\dataset\BirdClef\paper_dataset\Checkpoint'

def get_geners(args, batch_size, Img_size):
    gener = Data_Gener(mode=args.mode, Img_size=Img_size)
    # TODO:Batch_size 应该也要能自己设置
    source_gen, source_num = gener.data_gener(data_file_path=args.source_path, BatchSize=batch_size, label_type='source',
                                              spec_path=args.source_spec_path, mode=args.mode)
    # x, y = next(source_gen)
    target_gen, target_num = gener.data_gener(data_file_path=args.target_path, BatchSize=batch_size, label_type='target',
                                              spec_path=args.target_spec_path, mode=args.mode, return_label=False)
    val_gen, val_num = gener.data_gener(data_file_path=args.validation_path, BatchSize=batch_size, label_type='source',
                                        spec_path=args.source_spec_path, mode=args.mode, aug=False)
    test_gen, test_num = gener.data_gener(data_file_path=args.validation_path, BatchSize=batch_size, label_type='source',
                                        spec_path=args.source_spec_path, mode=args.mode, aug=False)
    return (gener, source_gen, source_num, target_gen, target_num, val_gen, val_num, test_gen, test_num)

if __name__ == '__main__':
    # TODO(2020/1/27): 完成深度学习下的source训练
    # TODO(2020/1/31): 完成深度学习下的整个流程
    # 命令行该怎么输入？:
    # python main_adda.py -参数1缩写 参数1输入 -参数2缩写 参数2输入

    ap = argparse.ArgumentParser()
    # 前面的 ‘-s’ 格式是缩写指代参数名
    # TODO:这里应该是在命令行中进行控制，应该选个方法是在idle中进行控制
    ap.add_argument('--mode', default='Species', help='The mode is to choose the classifying range')
    ap.add_argument('-lps', '--source_label_path', default=r'G:\dataset\BirdClef\vacation\source_check.csv')
    ap.add_argument('-lpt', '--target_label_path', default=r'G:\dataset\BirdClef\vacation\target_check.csv')
    ap.add_argument('-sp', '--source_path', default=r'G:\dataset\BirdClef\vacation\train_file\source')
    ap.add_argument('-tp', '--target_path', default=r'G:\dataset\BirdClef\vacation\train_file\target')
    ap.add_argument('-SSP', '--source_spec_path', default=r'G:\dataset\BirdClef\vacation\spectrum\source')
    ap.add_argument('-TSP', '--target_spec_path', default=r'G:\dataset\BirdClef\vacation\spectrum\target')
    ap.add_argument('--batch_size', default=16, type=int, help='The input size that every iteration go through in.')
    args = ap.parse_args()
    # TODO:模型不同，输入的label_len应该不同，这里先用20替代

    # Get the data generator
    source_gener = Data_Gener(mode=args.mode, Img_size=[256, 512],
                              label_path=args.source_label_path)
    [train_source, val_source, test_source], lens = source_gener.data_gener(data_file_path=args.source_path,
                                                                            BatchSize=16,
                                                                            spec_path=args.source_spec_path)

    target_gener = Data_Gener(mode=args.mode, Img_size=[256, 512],
                              label_path=args.target_label_path)
    [train_target, val_target, test_target], lens = target_gener.data_gener(data_file_path=args.target_path,
                                                                            BatchSize=8,
                                                                            spec_path=args.target_spec_path)

    train_xs, train_ys = next(val_source)
    train_xs = train_xs.reshape([train_xs.shape[0], -1])
    train_xt, train_yt = next(val_target)
    train_xt = train_xt.reshape([train_xt.shape[0], -1])


    tca = TCA(kernel_type='linear', dim=2048, lamb=1, gamma=1)
    xsnew, xtnew = tca.fit(train_xs, train_xt)

    mmd_b = mmd_linear(train_xs[:8], train_xt)
    mmd_a = mmd_linear(xsnew[:8], xtnew)