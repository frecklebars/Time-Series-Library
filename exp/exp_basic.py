import os
import torch
from torch import optim
import numpy as np

from utils.tools import adjust_learning_rate, EarlyStopping

from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, TemporalFusionTransformer, SCINet, \
    MambaSimple, Mamba, S_Mamba

from pretraining import ContrastiveBasic


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
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        self.pretrain_strategy_dict = {
            "ContrastiveBasic": ContrastiveBasic,
        }

        if self.args.is_pretraining:
            self.pretraining_strategy = self._build_pretrain_strategy()

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

    # pretrain
    def _build_pretrain_strategy(self):
        return self.pretrain_strategy_dict[self.args.pretrain_strategy].PreStrategy(self)

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def pretrain(self, setting):
        self.pretraining_strategy.pretrain()

    # def pretrain(self, setting):
    #     train_data, train_loader = self._get_data(flag='train')
    #     vali_data, vali_loader = self._get_data(flag='val')
    #     # TO-DO ! custom pretraining data provider
    #     # test_data, test_loader = self._get_data(flag='test')

    #     # data_loaders = [train_loader, vali_loader, test_loader]

    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    
    #     time_now = time.time()

    #     train_steps = len(train_loader)
    #     early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    #     model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    #     criterion = NTXentLoss(temperature=0.10)
    #     vali_criterion = self._select_criterion()

    #     # if self.args.use_amp:
    #     #     scaler = torch.cuda.amp.GradScaler()

    #     epoch_losses = []
    #     vali_losses = []

    #     for epoch in range(self.args.pretrain_epochs):
    #         iter_count = 0
    #         train_loss = []

    #         self.model.train()
    #         epoch_time = time.time()
                
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    #             iter_count += 1
                
    #             if (i + 1) % self.args.pre_accumulation_steps == 0 or i == train_steps - 1:
    #                 model_optim.zero_grad()

    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # TO-DO do the amp here

    #             batch_x_aug_1, batch_x_aug_2 = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

    #             batch_x_aug_1 = batch_x_aug_1.to(self.device)
    #             batch_x_aug_2 = batch_x_aug_2.to(self.device)
                
    #             outputs = torch.cat((batch_x_aug_1, batch_x_aug_2), dim=0)
    #             indices = torch.arange(0, batch_x_aug_1.shape[0], device=self.device)
    #             labels = torch.cat((indices, indices), dim=0)

    #             loss = criterion(outputs, labels)
    #             train_loss.append(loss.item())

    #             # if (i + 1) % 100 == 0:
    #             if (i + 1) % 60 == 0:
    #                 print("\titers: {0}, epoch{1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                 speed = (time.time() - time_now) / iter_count
    #                 left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #                 print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                 iter_count = 0
    #                 time_now = time.time()
                
    #             loss = loss / self.args.pre_accumulation_steps
    #             loss.backward()

    #             if (i + 1) % self.args.pre_accumulation_steps == 0 or i == train_steps - 1:
    #                 model_optim.step()
                
            
    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         train_loss = np.average(train_loss)
    #         epoch_losses.append(train_loss)
    #         vali_loss = self.vali(train_loader, vali_loader, vali_criterion)
    #         vali_losses.append(vali_loss)
                
    #         print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}, Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))
    #         early_stopping(vali_loss, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #         adjust_learning_rate(model_optim, epoch, self.args)

    #     print("Epoch Losses: ", epoch_losses)
    #     print("Vali  Losses: ", vali_losses)
    #     best_model_path = path + "/" + "checkpoint.pth"
    #     self.model.load_state_dict(torch.load(best_model_path))

    #     self.model.is_pretraining = False
    #     # self.args.learning_rate = self.args.ft_lr

    #     return self.model



