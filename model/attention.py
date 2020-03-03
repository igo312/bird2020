# Use numpy to make a program that can clip the audio (Unsupervised)
# This algorithm can detect the part of audio which has the biggest energy per unit
# Loss function will be the sum of energy and length

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage  as Image
import time

class SinglePerceptron(object):
    def __init__(self, lr_rate, alpha=0.3):
        '''
        :param lr_rate: learning rate
        :param alpha: a superparameter for scaling the function
        '''
        self.lr_rate = lr_rate
        self.alpha = alpha

    def weight_initial(self, shape):
        return np.random.randn(shape[0], shape[1])

    def create_loss(self, input, label):
        y_pre = np.dot(input, self.W)
        return 1/2 * np.sum(np.square(np.subtract(y_pre, label)))

    def create_output(self, input):
        return np.dot(input, self.W)

    def create_easiest_model(self, input_shape):
        self.W = self.weight_initial([input_shape[-1], 4])
        return

    def model_train(self, input, label):
        # L = 1/2 * norm2(y_pre - y_true)^2 = 1/2 * norm2(XW - y_true)^2
        # round L/ round W = (y_pre - y_true)T * X
        # W = W - lr * round L/ round W
        y_pre = self.create_output(input)
        round = np.dot(
            (y_pre-label).T,
            input
        )
        self.W = self.W - self.lr_rate * (round.T)
        return

class MultiPerceptron(object):
    def __init__(self,lr,batch_size):
        self.lr = lr
        self.batch_size = batch_size

    def model_initialize(self,input_shape):
        # Initialize the weights
        W1, b1 = np.random.randn(input_shape[-1], 512),  np.random.randn(input_shape[0], 512)
        W2, b2 = np.random.rand(W1.shape[1], 128), np.random.randn(input_shape[0], 128)
        W3, b3 = np.random.randn(W2.shape[1], 4), np.random.randn(input_shape[0], 4)

        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3
        return

    def layer_creative(self, input, W, b):
        return np.add(np.dot(input, W), b)

    def relu_act(self, input):
        input[np.where(input<0)] = 0
        return input

    def output_creative(self,input):
        l1 = self.layer_creative(input, self.W1, self.b1)
        l1_ = self.relu_act(l1)
        l2 = self.layer_creative(l1, self.W2, self.b2)
        l2_ = self.relu_act(l2)
        l3 = self.layer_creative(l2_, self.W3, self.b3)
        output = l3.copy()
        return output

    def loss_creative(self, input, label):
        y_pre = self.output_creative(input)
        return 1/2 * np.sum(np.square(np.subtract(y_pre, label)))

    def model_train(self, input, label):
        l1 = self.layer_creative(input, self.W1, self.b1)
        l1_ = self.relu_act(l1)
        l2 = self.layer_creative(l1, self.W2, self.b2)
        l2_ = self.relu_act(l2)
        l3 = self.layer_creative(l2_, self.W3, self.b3)
        y_pre = l3

        l2_round = np.zeros(l2_.shape)
        l2_round[np.where(l2_ > 0)] = 1

        l1_round = np.zeros(l1_.shape)
        l1_round[np.where(l1_ > 0)] = 1

        round_W3 = np.dot(l2_, (y_pre - label))
        round_b3 = y_pre - label
        round_W2 = l2_round




def image_read(path):
    img = plt.imread(path)
    img_ = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_.reshape([1,-1])

if __name__ == '__main__':
    # Use spectrum as dataset. Later will use temporal data
    # Cause it's a basic attetion model, I use perceptron which only allow one dimention
    train_data = image_read(r'F:\鸟鸣\较好音频.jpg')
    test_data = image_read(r'F:\鸟鸣\背景大于鸟声.jpg')
    label = [2085.36, 1864.63, 474.82, 397.27]

    # input的输入太多，XW输出就会很大，因为W初始化做的不好，因此loss就会特别大。
    # loss大的结果是 round L / round y的值也特别大，若学习率在0.001左右，W会被更新的十分快，出现了梯度爆炸，瞬间不收敛。
    model = ClipModel(lr_rate=0.0000000000005)
    model.create_easiest_model(train_data.shape)
    init_loss = model.create_loss(train_data, label)

    # 训练过程
    print('开始训练')
    i= 0
    while i < 220:
        model.model_train(train_data, label)
        i += 1
    print('训练完毕')

    time.sleep(1)
    result_loss = model.create_loss(train_data, label)
    output = model.create_output(train_data)
    print('预测的坐标为', output)
    print('初始损失为{:.2f}, 训练后的损失为{:.2f}'.format(init_loss, result_loss))

    plt.figure()
    img = plt.imread(r'F:\鸟鸣\较好音频.jpg')
    output = output.squeeze(axis=0)

    plt.subplot(211)
    plt.imshow(img)

    #打标签
    current_axis = plt.gca()

    current_axis.add_patch(
        plt.Rectangle((label[1], label[3]), label[0] - label[1], label[2] - label[3], color='green', fill=False, linewidth=2)
    )
    current_axis.text(label[1], label[3], 'true_ground', fontsize=5, color='white',
                      bbox={'facecolor': 'green', 'alpha': 1.0})

    plt.subplot(212)
    plt.imshow(img)
    current_axis = plt.gca()
    current_axis.add_patch(
        plt.Rectangle((output[1], output[3]), output[0] - output[1], output[2] - output[3], color='red', fill=False,linewidth=2)
    )
    current_axis.text(output[1], output[3], 'pre_ground', fontsize=5, color='white',
                      bbox={'facecolor': 'red', 'alpha': 1.0})
    plt.show()
    #img = plt.imread(r'F:\鸟鸣\较好音频.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

