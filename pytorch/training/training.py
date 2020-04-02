# TODO:哪些值需要 to(gpu)?
# TODO:还没做evaluate的模型
# TODO:batchnorm2d是啥玩意？
# TODO:training 数据generator还没好

import argparse
import csv
import os
import random
import sys
sys.path.append(r'f:\pycharm\vacation\pytorch')

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from tqdm import trange

from pytorch.model.mobilenetv2 import MobileNet2
from pytorch.model.mobilenetv1 import MobileNet1
from pytorch.utils import TrainMethod, logger_create, save_checkpoint
from Preprocessing.generator.generator_for_MCCNN import generator
import datetime
from tensorboardX import SummaryWriter
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch training processing')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')

# Checkpoints
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
parser.add_argument('--weight', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')

# Architecture
parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of MobileNet (default x1).')
parser.add_argument('--img-size', type=int, default=[256,256,3], metavar='I',
                    help='Input size of MobileNet, multiple of 32 (default 256).')
parser.add_argument('--input-size', type=int, default=[1,3,256,256], metavar='I',
                    help='Input size of MobileNet, multiple of 32 (default 224).')

# Training
parser.add_argument('--start_epoch', type=int, default=0, help='The start index of model training')
parser.add_argument('--epochs', type=int, default=128, help='The total training epochs in one time')
parser.add_argument('--model_name', type=str, help='It is for some saving name')
parser.add_argument('-lps', '--label_path', default=r'G:\dataset\BirdClef\vacation\limit_species.csv')
parser.add_argument('--class_num', default=10,help='Decide the class num of model will yiled')
parser.add_argument('--spec_path', help='The spectrum generator path')
parser.add_argument('--train_mode', help='mode decide the model will learn or do evaluation')

# Optimization options
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[30,55,80,100,110],
                    help='Decrease learning rate at these epochs.')

if __name__ == '__main__':
    # -----------------------------------------------model setting------------------------------------------------------#

    args = parser.parse_args()
    args.gpus = ['cuda']
    args.save = r'G:\dataset\BirdClef\vacation\Checkpoint'
    args.train_mode = True

    args.model_name = 'mobV1S1_30'
    args.class_num = 30
    # args.weight = r'G:\dataset\BirdClef\vacation\Checkpoint\mobV1S1-epoch69-val_loss1.1361.pth'
    args.spec_path = r'G:\dataset\BirdClef\vacation\spectrum\ICA30S1'
    args.label_path = r'G:\dataset\BirdClef\vacation\lssource_limit30.csv'
    args.img_size = [256,256,3]
    args.input_size = [1,3,256,256]

    if args.weight:
        print('Weight loaded')
        # must set map_location
        weight = torch.load(args.weight, map_location=args.gpus[0])
        args.start_epoch = weight['epoch']
    else:
        print('training a new model')

    #if args.seed is None:
    #    args.seed = random.randint(1, 10000)
    args.seed = 5153
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        # Sets the seed for generating random numbers on all GPUs.
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.gpus is not None:
        #args.gpus = [int(i) for i in args.gpus.split(',')]
        #device = 'cuda:' + str(args.gpus[0])
        #TODO(3/30):这里有点问题，暂时只用cuda
        #device = ['cuda:' + str(i) for i in args.gpus]
        device = 'cuda'
        # 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销，一般都会加。
        cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    model = MobileNet1(args.img_size[-1], args.class_num)
    #model = MobileNet2(input_size=args.img_size, scale=args.scaling, num_classes=args.class_num)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    # 需要转移到最终的device上再进行load，否则会报错
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)
    if args.weight:
        model.load_state_dict(weight['state_dict'])
        optimizer.load_state_dict(weight['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.weight, weight['epoch']))
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=args.gamma)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    # print(model)
    print('number of parameters: {}'.format(num_parameters))

    #------------------------------------------data generator loading--------------------------------------------------#
    # if args.evaluate:
    #    loss, top1, top5 = test(model, val_loader, criterion, device, dtype)  # TODO
    #    return
    train = generator(args.img_size, args.label_path)
    train_num, train = train(os.path.join(args.spec_path,'train'), args.batch_size, True)
    val = generator(args.img_size, args.label_path)
    val_num, val = val(os.path.join(args.spec_path,'validation'), args.batch_size, False)
    test = generator(args.img_size, args.label_path)
    test_num, test = test(os.path.join(args.spec_path,'test'), args.batch_size, False)
    #------------------------------------------------------------------------------------------------------------------#

    #-----------------------------------------------training processing-------------------------------------------------#
    best_test = 0
    date = datetime.datetime.strftime(datetime.datetime.now(), '%m%d%H%M')
    #logger = logger_create(name=date+'-'+args.model_name,
    #                       path=r'G:\dataset\BirdClef\vacation\torch-{}-{}.log'.format(args.model_name, date))
    if args.train_mode:
        train_writer = SummaryWriter(r'G:\dataset\BirdClef\vacation\Checkpoint\tensorboard\{}\train'.format('pytorch-'+args.model_name), )
        val_writer = SummaryWriter(r'G:\dataset\BirdClef\vacation\Checkpoint\tensorboard\{}\validation'.format('pytorch-'+args.model_name), )
        dummy_input = torch.rand(args.input_size).to(device)
        train_writer.add_graph(model,(dummy_input,))
        val_writer.add_graph(model, (dummy_input,))
        for epoch in range(args.start_epoch, args.epochs):
            train_loss, train_accuracy1 = TrainMethod.train(model, train, epoch, train_num//args.batch_size, optimizer, criterion, device,
                                                                  dtype, log_interval=5, batch_size=args.batch_size, img_mode='NHWC')
            val_loss, val_accuracy1,  = TrainMethod.test(model, val, 512//args.batch_size, criterion, device, dtype, img_mode='NHWC')

            val_writer.add_scalar('epoch_accuracy', val_accuracy1, epoch)
            val_writer.add_scalar('epoch_loss', val_loss, epoch )
            train_writer.add_scalar('epoch_loss', train_loss, epoch)
            train_writer.add_scalar('epoch_accuracy', train_accuracy1, epoch)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif isinstance(scheduler, MultiStepLR):
                scheduler.step()

            if val_accuracy1 > best_test:
                best_test = val_accuracy1
            # TODO:没输入进去？
            # logger.info('epoch:{}, val_loss:{:.5f}, val_accuracy:{:.5f}'.format(epoch+1, val_loss, val_accuracy1))
            save_checkpoint(
                            is_best=False, filepath=args.save,
                            filename='{}-epoch{}-val_loss{:.4f}.pth'.format(args.model_name, epoch, val_loss),
                            state={'epoch': epoch , 'state_dict': model.state_dict(), 'best_prec1': best_test,
                                    'optimizer': optimizer.state_dict()},
            )
    else:
        print('Doing evaluation')
        test_loss, test_accuracy1, = TrainMethod.test(model, test, test_num // args.batch_size, criterion, device, dtype,
                                                    img_mode='NHWC')

# x = np.load(r'G:\dataset\BirdClef\vacation\testS1x.npy')
# y= np.load(r'G:\dataset\BirdClef\vacation\testS1y.npy')
# r2 = model(torch.from_numpy(x).cuda())
# r2_ = r2.cpu().detach().numpy()
# r1 = np.load(r'G:\dataset\BirdClef\vacation\r1.npy')
# r2 = np.load(r'G:\dataset\BirdClef\vacation\r2.npy')