import os
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.esim import ESIM
from sampler.sampler import Sampler
from utils.macros import *


class Solver:
    def __init__(self, model, n_epochs=1000, lr=1e-4,
                 print_every=100, save_every=100, save_path=SAVE_PATH):
        """
        :param model: 模型
        :param n_epochs: 训练周期
        :param lr: 学习率
        :param print_every: 每训练多少轮打印一次报告
        :param save_every: 每训练多少轮存储一次模型
        :param save_path: 模型的存储路径
        """
        self.sampler = Sampler()
        self.model = model
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.save_every = save_every
        self.save_path = save_path
        if CUDA_AVAILABLE:
            self.model = self.model.cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        self.crit = nn.BCELoss()

        if os.path.exists(self.save_path):
            self.model = torch.load(self.save_path)

    def solve(self):
        """
        训练模型, solve 的意思是解方程, 具体而言, 解的是模型的参数
        """
        print('Start training...')
        self.model.train()
        for epoch in range(1, self.n_epochs+1):
            batch_x1, batch_x2, batch_y = self.sampler.next_batch()
            batch_x1 = torch.from_numpy(batch_x1.astype(np.int64)).to(device)
            batch_x2 = torch.from_numpy(batch_x2.astype(np.int64)).to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            batch_y_hat = self.model((batch_x1, batch_x2))
            loss = self.crit(batch_y_hat, batch_y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if epoch % self.print_every == 0 or epoch == self.n_epochs:
                print('Training %d/%d - Loss:%.3f ...' %
                      (epoch, self.n_epochs, torch.sum(loss.data)/batch_x.size()[0]))
            if epoch % self.save_every == 0 or epoch == self.n_epochs:
                torch.save(self.model, self.save_path)
                print('Model saved')
        print('Training finished!')

    def evaluate(self):
        """
        验证模型在测试集上的性能
        """
        print('Start evaluating...')
        self.model.eval()
        # 获取测试集
        test_x, test_y = self.sampler.test_set()
        test_x = torch.FloatTensor(test_x).to(device)
        # 预测
        pred_y = (self.model.forward(test_x).data.numpy() >= 0.5).astype(np.int32)
        # 获取准确率
        accuracy = np.sum(pred_y == test_y) / test_y.shape[0]
        print('Test accuracy: %.3f' % accuracy)

