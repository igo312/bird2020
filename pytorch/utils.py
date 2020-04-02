import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange

matplotlib.use('Agg')

from matplotlib import pyplot as plt
import logging
import numpy as np

class TrainMethod(object):
    @staticmethod
    def train(model, loader, epoch, training_steps, optimizer, criterion, device, dtype, batch_size, log_interval, img_mode, accumulate_step=1,one_hot=True):
        model.train()
        correct1, correct5 = 0, 0
        corr_interval = 0
        loss_t = 0
        loss_temp = 0
        optimizer.zero_grad()
        for batch_idx in trange(training_steps):
            #if isinstance(scheduler, CyclicLR):
            #    scheduler.batch_step()
            # TODO(3/30):pytorch怎么做prefetch？
            if one_hot:
                data, target = TrainMethod._next(loader, device, dtype, img_mode)
            else:
                raise ValueError('Should check the script to load data whose one-hot is FALSE')

            # 每次loss.backward()都会积累梯度
            # optimizer.zero_grad()清除积累的梯度
            # optimizer.step()以积累的梯度更新参数
            output = model(data)
            loss = criterion(output, target)
            loss /= accumulate_step
            loss.backward()
            # Doing accumu step
            if (batch_idx+1) % accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            corr = TrainMethod.correct(output, target, topk=(1,))
            corr_interval += corr[0]
            correct1 += corr[0]
            # correct5 += corr[1]

            # loss_temp 用于log_inteval时候的阶段性输出，loss_t用于整个epoch训练过后的输出
            l = loss.item()
            loss_temp += l
            loss_t += l

            if batch_idx % log_interval == 0:
                tqdm.write(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. training accuracy: batch_acc:{:.2f}%(total:acc:{:.2f}%).'.format(
                                                               epoch, batch_idx, training_steps,100. * batch_idx / training_steps,
                                                               loss_temp / log_interval,
                                                               100. * corr_interval  / batch_size / log_interval,
                                                               100. * correct1 / (batch_size * (batch_idx + 1)),
                                                                ))
                corr_interval = 0
                loss_temp = 0
        return loss_t / training_steps, correct1 / training_steps/ batch_size

    @staticmethod
    def test(model, loader, test_steps, criterion, device, dtype, img_mode):
        model.eval()
        test_loss = 0
        correct1, correct5 = 0, 0

        for _ in range(test_steps):
            data, target = TrainMethod._next(loader, device, dtype, img_mode)
            with torch.no_grad():
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                corr = TrainMethod.correct(output, target, topk=(1, ))
            # correct 记录了正确的数目，除以data.shape[0]转换为准确率
            correct1 += corr[0] / data.shape[0]


        test_loss /= test_steps

        tqdm.write(
            ('\nTest set: Average loss: {:.4f}, accuracy: {}/{} ({:.2f}%), ').format(
                                           test_loss, int(correct1), test_steps,
                                           100. * correct1 / test_steps,
                                           ))
        return test_loss, correct1 / test_steps

    @staticmethod
    def _next(loader, device, dtype, img_mode):
        data, target = next(loader)
        target = np.argmax(target, axis=1)
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        if img_mode == 'NCHW':
            pass
        elif img_mode == 'NHWC':
            data = data.permute([0, 3, 1, 2])
        else:
            raise ValueError('Img channel order must be "NCHW" or "NHWC"')
        data, target = data.to(device=device, dtype=dtype), target.to(device=device, dtype=torch.long)
        return data, target

    @staticmethod
    def correct(output, target, topk=(1,)):
        # top1返回正确的个数，不是准确率
        # TODO(3/31)：掌握top1，top5的写法
        """Computes the correct@k for the specified values of k"""
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().type_as(target)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0).item()
            res.append(correct_k)
        return res

    @staticmethod
    def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
        save_path = os.path.join(filepath, filename)
        best_path = os.path.join(filepath, 'model_best.pth.tar')
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, best_path)

    @staticmethod
    def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6, max_lr=8e-5, step_size=2000,
                        mode='triangular', save_path='.'):
        model.train()
        correct1, correct5 = 0, 0
        scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
        epoch_count = step_size // len(loader)  # Assuming step_size is multiple of batch per epoch
        accuracy = []
        for _ in trange(epoch_count):
            for batch_idx, (data, target) in enumerate(tqdm(loader)):
                if scheduler is not None:
                    scheduler.batch_step()
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                corr = TrainMethod.correct(output, target)
                accuracy.append(corr[0] / data.shape[0])

        lrs = np.linspace(min_lr, max_lr, step_size)
        plt.plot(lrs, accuracy)
        plt.show()
        plt.savefig(os.path.join(save_path, 'find_bounds_clr.png'))
        np.save(os.path.join(save_path, 'acc.npy'), accuracy)
        return

def logger_create(name, path):
    mylogger = logging.getLogger(name)
    log_path = path
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    logging.basicConfig(level=logging.DEBUG,
                        format=formatter
                        )
    fh.setFormatter(formatter)
    mylogger.addHandler(fh)
    return mylogger

def save_checkpoint(state, is_best, filename, filepath='./', ):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)

