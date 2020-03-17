# -*- encoding -*- utf-8
from __future__ import print_function, division
# TODO：所以是用了多个keras.model，利用output进行复用？
from tensorflow import keras

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Lambda, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, GlobalAvgPool2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, clone_model
from keras.optimizers import Adam
from keras import regularizers
# TODO：学习np_utils
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf


import matplotlib.pyplot as plt
#TODO:for layer in self.source_encoder.layers:
#      layer.trainable = False
import warnings
import sys
import os
import numpy as np
import argparse
save_path = r'G:\dataset\BirdClef\vacation\Checkpoint'

class ADDA():
    def __init__(self, lr, clf_mode, class_num, Img_size):
        # Input shape
        self.img_rows, self.img_cols = Img_size[0], Img_size[1]
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.src_flag = False
        self.disc_flag = False

        self.discriminator_decay_rate = 5

        # iterations
        self.discriminator_decay_factor = 0.5
        # src:source, tgt:target
        self.src_optimizer = Adam(lr, 0.5)
        self.tgt_optimizer = Adam(lr, 0.5)

        # multi layer classify num dict
        self.clf_mode = clf_mode
        self.clf_dict = {
            'Family': None,
            'Genus': None,
            'Species': 30
                    }
        #self.clf_num = self.clf_dict[clf_mode]
        self.clf_num = class_num

    def BN_CONV(self, x, FNum, FSize, strides=(1, 1)):
        # The activation is relu default(it will add other activation in future)
        x = Conv2D(FNum, FSize, padding='same', strides=strides)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def BN_FC(self, x, FNum):
        x = Dense(FNum)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def res_block(self, x, FNum, FSize, strides=(1, 1)):
        if strides == (2,2):
            x2 = MaxPooling2D()(x)
        elif strides == (1,1):
            x2 = x
        else:
            raise ValueError('Strides only can be (1,1) or (2,2)')
        x1 = self.BN_CONV(x, FNum, FSize, strides=strides)
        x = concatenate([x2, x1])
        return self.BN_CONV(x, FNum, 1)

    def define_source_encoder(self, weights=None):
        # TODO：这里是不是重复定义了self.source_encoder
        # TODO(11/11):可以用自己的模型，但现在主要先把整个流程搭建好
        self.source_encoder = Sequential()
        inp = Input(shape=self.img_shape)
        #TODO : when debug the batchnorm will stop the code
        # 让初始层学习更多的粗略特征，用于保留
        #x = BatchNormalization()(inp)
        x = self.BN_CONV(inp, 64, 3, strides=(2,2))
        x = MaxPooling2D()(x)
        x = self.BN_CONV(x, 128, 3, strides=(2, 2))
        x = MaxPooling2D()(x)
        x = self.res_block(x, 256, 3, strides=(2, 2))
        # x = self.BN_CONV(x, 256, 3)
        x = MaxPooling2D()(x)
        x = self.res_block(x, 512, 3,)
        x = MaxPooling2D()(x)
        #x = self.res_block(x, 361, 3)
        #x = MaxPooling2D()(x)
        self.source_encoder = Model(inputs=(inp), outputs=(x))
        self.src_flag = True

        if weights is not None:
            # TODO:by_name 选项是什么要知道，load_weights函数 ->
            # 从HDF5文件中加载权重到当前模型中, 默认情况下模型的结构将保持不变。\
            # 如果想将权重载入不同的模型（有些层相同）中，则设置by_name=True，只有名字匹配的层才会载入权重
            self.source_encoder.load_weights(weights, by_name=True)
            # self.source_encoder.load_weights(weights)

    def get_source_classifier(self, model, weights=None, mode=None):
        # model已经经过编译了，只是采用了相应的input和output
        #x = GlobalAvgPool2D()(model.output)
        # Use globalAvgPool2D will cut off 3000000 (The total is 7200000)
        x = Flatten()(model.output)
        x = self.BN_FC(x, 128)
        x = Dropout(0.5)(x)
        if mode == None:
            print('The classify num is {}'.format(self.clf_num))
            x = Dense(self.clf_num, activation='softmax')(x)
        else:
            x = Dense(self.clf_dict[mode], activation='softmax')(x)
        source_classifier_model = Model(inputs=(model.input), outputs=(x), name=self.clf_mode+'_clf_model')

        if weights is not None:
            print('Loading pre-trained classifier model')
            source_classifier_model.load_weights(weights)
        return source_classifier_model

    def define_target_encoder(self, weights=None):
        # src_flag：应该是用来判断是否生成模型，如果false则生成模型
        if not self.src_flag:
            self.define_source_encoder()
        # TODO:去看看clone_model这个函数怎么用
        # Clone any `Model` instance.
        with tf.device('/cpu:0'):
            self.target_encoder = clone_model(self.source_encoder)

        if weights is not None:
            self.target_encoder.load_weights(weights, by_name=True)

    def _define_discriminator(self, shape):
        inp = Input(shape=shape)
        x = GlobalAvgPool2D()(inp)
        # TODO:what is kernel_regularizer?
        # kernel_regularizer：初看似乎有点费解，kernel代表什么呢？其实在旧版本的Keras中，该参数叫做weight_regularizer，\
        # 即是对该层中的权值进行正则化，亦即对权值进行限制，使其不至于过大。
        x = Dense(128, activation=LeakyReLU(alpha=0.3), kernel_regularizer=regularizers.l2(0.01),
                  name='discriminator1')(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='sigmoid', name='discriminator2')(x)

        self.disc_flag = True
        self.discriminator_model = Model(inputs=(inp), outputs=(x), name='discriminator')

    def get_discriminator(self, model, weights=None):

        if not self.disc_flag:
            self._define_discriminator(model.output_shape[1:])

        disc = Model(inputs=(model.input), outputs=(self.discriminator_model(model.output)))

        if weights is not None:
            disc.load_weights(weights, by_name=True)

        return disc

    def tensorboard_log(self, callback, names, logs, batch_no):

        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

    def train_source_model(self, source_gen, val_gen, model, epochs, steps_per_epoch, validation_steps, name,
                           save_interval=10, start_epoch=0, lr=0.01, C_state=True):
        optimizer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # ModelCheckpoint 回调类允许你定义检查模型权重的位置，文件应如何命名，以及在什么情况下创建模型的 Checkpoint。
        saver = keras.callbacks.ModelCheckpoint(os.path.join(save_path, name+'{epoch:02d}-{val_acc:.2f}.hdf5'),
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=False,
                                                save_weights_only=True,
                                                mode='auto',
                                                period=save_interval)

        # 用于自动调节学习率，factor代表衰减因子
        scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=0,
                                                      mode='min')

        # 用于提前终止训练防止过拟合
        early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tensorboard_save = os.path.join(save_path, 'tensorboard'+'\\'+name)
        if not os.path.isdir(tensorboard_save):
            os.mkdir(tensorboard_save)

        visualizer = keras.callbacks.TensorBoard(log_dir=tensorboard_save,
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=False)
        #TODO:为什么一个fit_generator会创建多个tensorboard
        if C_state:
            callbacks = [saver, scheduler, visualizer]
        else:
            callbacks = []
        model.fit_generator(source_gen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_gen,
                            validation_steps=validation_steps,
                            initial_epoch=start_epoch,
                            )

    def train_target_discriminator(self, source_gen, target_gen, source_model=None, src_discriminator=None,
                                   tgt_discriminator=None, epochs=50, save_interval=1, start_epoch=0, num_batches=200):
        '''
        :param batch_data:
        :param source_model:
        :param src_discriminator:
        :param tgt_discriminator:
        :param epochs:
        :param batch_size:
        :param save_interval:
        :param start_epoch:
        :param num_batches:  一个epoch所循环的次数
        :return:
        '''
        # TODO：从这里是不是可以看出，不用一对一的指定标签？
        self.define_source_encoder(source_model)

        # TODO：起到了freeze的功能？
        for layer in self.source_encoder.layers:
            layer.trainable = False

        # get_discriminator(self, model, weights=None):
        source_discriminator = self.get_discriminator(self.source_encoder, src_discriminator)
        target_discriminator = self.get_discriminator(self.target_encoder, tgt_discriminator)
        # TODO:这里是不是和和 self.get_discriminator函数重复了加载功能？
        '''
        if src_discriminator is not None:
            source_discriminator.load_weights(src_discriminator)
        if tgt_discriminator is not None:
            target_discriminator.load_weights(tgt_discriminator)
        '''

        # TODO：为什么使用了binary_crossentropy？没有label输入啊？ -> 后面有输入
        source_discriminator.compile(loss="binary_crossentropy", optimizer=self.tgt_optimizer, metrics=['accuracy'])
        target_discriminator.compile(loss="binary_crossentropy", optimizer=self.tgt_optimizer, metrics=['accuracy'])

        # TODO(11/12):更改路径
        callback1 = keras.callbacks.TensorBoard(os.path.join(save_path,'tensorboard','binary'))
        callback1.set_model(source_discriminator)
        callback2 = keras.callbacks.TensorBoard(os.path.join(save_path,'tensorboard','binary'))
        callback2.set_model(target_discriminator)
        src_names = ['src_discriminator_loss', 'src_discriminator_acc']
        tgt_names = ['tgt_discriminator_loss', 'tgt_discriminator_acc']

        for iteration in range(start_epoch, epochs):

            avg_loss, avg_acc, index = [0, 0], [0, 0], 0
            # TODO：用这种想法实现了discriminator的loss，因为前几层都是共享的
            # source_gen -> use function(next()) get the tuple (img, label)
            for source, target in zip(next(source_gen)[0], next(target_gen)[0]):
                l1, acc1 = source_discriminator.train_on_batch(source,
                                                               np_utils.to_categorical(np.zeros(source.shape[0]), 2))
                l2, acc2 = target_discriminator.train_on_batch(target, np_utils.to_categorical(np.ones(target.shape[0]), 2))
                index += 1
                loss, acc = (l1 + l2) / 2, (acc1 + acc2) / 2
                print(iteration + 1, ': ', index, '/', num_batches, '; Loss: %.4f' % loss, ' (', '%.4f' % l1,
                      '%.4f' % l2, '); Accuracy: ', acc, ' (', '%.4f' % acc1, '%.4f' % acc2, ')')
                avg_loss[0] += l1
                avg_acc[0] += acc1
                avg_loss[1] += l2
                avg_acc[1] += acc2
                if index % num_batches == 0:
                    break

            if iteration % self.discriminator_decay_rate == 0:
                lr = K.get_value(source_discriminator.optimizer.lr)
                K.set_value(source_discriminator.optimizer.lr, lr * self.discriminator_decay_factor)
                lr = K.get_value(target_discriminator.optimizer.lr)
                K.set_value(target_discriminator.optimizer.lr, lr * self.discriminator_decay_factor)
                print('Learning Rate Decayed to: ', K.get_value(target_discriminator.optimizer.lr))
            # TODO(11/12)：从这里修改地址，修改权重名称
            if iteration % save_interval == 0:
                source_discriminator.save_weights(os.path.join(save_path,'discriminator_source_%02d.hdf5' % iteration))
                target_discriminator.save_weights(os.path.join(save_path,'discriminator_target_%02d.hdf5' % iteration))

            self.tensorboard_log(callback1, src_names, [avg_loss[0] / source.shape[0], avg_acc[0] / source.shape[0]],
                                 iteration)
            self.tensorboard_log(callback2, tgt_names, [avg_loss[1] / target.shape[0], avg_acc[1] / target.shape[0]],
                                 iteration)

    def eval_source_classifier(self, test_gen, model, dataset='target',  domain='Source'):
        # TODO: target 和 source数据划分怎么分呢？
        model.compile(loss='categorical_crossentropy', optimizer=self.src_optimizer, metrics=['accuracy'])
        scores = model.evaluate_generator(test_gen, 100)
        print('Classifier Test loss:%.5f' % (scores[0]))
        print('Classifier Test accuracy:%.2f%%' % (float(scores[1]) * 100))
        #scores = model.evaluate_generator(test_gen, 100)
        #print('%s %s Classifier Test loss:%.5f' % (dataset.upper(), domain, scores[0]))
        #print('%s %s Classifier Test accuracy:%.2f%%' % (dataset.upper(), domain, float(scores[1]) * 100))

    def eval_target_classifier(self, source_model, target_discriminator, dataset='svhn'):

        self.define_target_encoder()
        model = self.get_source_classifier(self.target_encoder, source_model)
        model.load_weights(target_discriminator, by_name=True)
        model.summary()
        self.eval_source_classifier(model, dataset=dataset, domain='Target')

def _blockA(inp, out_channel):
    if out_channel > 4*96:
        warnings.warn('The output is larger than origin output')
    #x1 = AveragePooling2D(strides=(1, 1))(inp)
    x1 = Flatten()(inp)
    x1 = Conv2D(96, 1, activation='relu')(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(64, 1, activation='relu')(inp)
    x2 = Conv2D(96, 3, activation='relu')(x2)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(64, 1, activation='relu')(inp)
    x3 = Conv2D(96, 3, activation='relu')(x3)
    x3 = Conv2D(96, 3, activation='relu')(x3)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(96, 1, activation='relu')(inp)
    x4 = BatchNormalization()(x4)

    add = Add()([inp, x1, x2, x3, x4])
    output = Conv2D(out_channel, 1, activation='relu')(add)
    return output

if __name__ == '__main__':

    # TODO：学习argparse这个模块
    ap = argparse.ArgumentParser()
    # TODO:前面的 ‘-s’ 格式是缩写指代参数名
    ap.add_argument('-s', '--source_weights', required=False,
                    help="Path to weights file to load source model for training classification/adaptation")
    ap.add_argument('-e', '--start_epoch', type=int, default=1, required=False,
                    help="Epoch to begin training source model from")
    ap.add_argument('-n', '--discriminator_epochs', type=int, default=10000,
                    help="Max number of steps to train discriminator")
    ap.add_argument('-l', '--lr', type=float, default=0.0001, help="Initial Learning Rate")
    ap.add_argument('-f', '--train_discriminator', action='store_true',
                    help="Train discriminator model (if TRUE) vs Train source classifier")
    ap.add_argument('-a', '--source_discriminator_weights', help="Path to weights file to load source discriminator")
    ap.add_argument('-b', '--target_discriminator_weights', help="Path to weights file to load target discriminator")
    ap.add_argument('-t', '--eval_source_classifier', default=None,
                    help="Path to source classifier model to test/evaluate")
    ap.add_argument('-d', '--eval_target_classifier', default=None,
                    help="Path to target discriminator model to test/evaluate")

    args = ap.parse_args()

    adda = ADDA(args.lr, 'Family')
    adda.define_source_encoder()
    model = adda.get_source_classifier(adda.source_encoder, args.source_weights)

    if not args.train_discriminator:
        if args.eval_source_classifier is None:
            model = adda.get_source_classifier(adda.source_encoder, args.source_weights)
            adda.train_source_model(model, start_epoch=args.start_epoch - 1)
        else:
            model = adda.get_source_classifier(adda.source_encoder, args.eval_source_classifier)
            adda.eval_source_classifier(model, 'mnist')
            adda.eval_source_classifier(model, 'svhn')
    adda.define_target_encoder(args.source_weights)

    if args.train_discriminator:
        adda.train_target_discriminator(epochs=args.discriminator_epochs,
                                        source_model=args.source_weights,
                                        src_discriminator=args.source_discriminator_weights,
                                        tgt_discriminator=args.target_discriminator_weights,
                                        start_epoch=args.start_epoch - 1)
    if args.eval_target_classifier is not None:
        adda.eval_target_classifier(args.eval_source_classifier, args.eval_target_classifier)