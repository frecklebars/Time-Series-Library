import os
import time

import torch
from torch import optim
from pytorch_metric_learning.losses import NTXentLoss
import numpy as np

from utils.tools import adjust_learning_rate, EarlyStopping

from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, Mamba, S_Mamba

from contrastive.augmentation import RandomAUG


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'Mamba': Mamba,
            "S_Mamba": S_Mamba,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def pretrain(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val') # TODO ! custom pretraining data provider
        # test_data, test_loader = self._get_data(flag='test')

        # data_loaders = [train_loader, vali_loader, test_loader]

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
    
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = NTXentLoss(temperature=0.10)

        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()

        epoch_losses = []

        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
                
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # TODO do the amp here

                batch_x_aug_1, batch_x_aug_2 = self.model.contrastive_pretrain(batch_x, batch_x_mark, batch_y, batch_y_mark)

                batch_x_aug_1 = batch_x_aug_1.to(self.device)
                batch_x_aug_2 = batch_x_aug_2.to(self.device)

                outputs = torch.cat((batch_x_aug_1, batch_x_aug_2), dim=0)
                indices = torch.arange(0, batch_x_aug_1.shape[0], device=self.device)
                labels = torch.cat((indices, indices), dim=0)

                loss = criterion(outputs, labels)
                train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                if (i + 1) % 60 == 0:
                    print("\titers: {0}, epoch{1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            epoch_losses.append(train_loss)
                
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch, self.args)

        print("Epoch Losses: ", epoch_losses)
        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        self.model.is_pretraining = False

        return self.model



