import os
import time

from utils.tools import adjust_learning_rate, EarlyStopping

from pytorch_metric_learning.losses import NTXentLoss
from torch import optim

from pretraining.PretrainStrategy import PretrainStrategy

'''
Basic augment then contrast
'''

class PreStrategy(PretrainStrategy):
    def __init__(self, exp):
        super(Strategy, self).__init__(exp)

    def _select_criterion(self):
        criterion = NTXentLoss(temperature=0.10)
        return criterion

    def _select_optimizer(self):
        model_optim = optim.Adam(self.experiment.model.parameters(), lr=self.experiment.args.learning_rate)
        return model_optim

    def pretrain(self):
        print("pretraining contrastive basic")
        pass

        pretrain_data, pretrain_loader = self.experiment._get_data(flag='train') # TODO custom pretraining data provider

        path = os.path.join(self.experiment.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(pretrain_loader)
        early_stopping = EarlyStopping(patience=self.experiment.args.patience, verbose=True)

        model_optim = self.experiment._select_optimizer()
        criterion = self._select_criterion()
        
        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()

        epoch_losses = []

        for epoch in range(self.experiment.args.pretrain_epochs):
            iter_count = 0
            train_loss = []

            self.experiment.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pretrain_loader):
                iter_count += 1

                if (i + 1) % self.args.accumulation_steps == 0 or i == train_steps - 1:
                    model_optim.zero_grad()

                # augment
                batch_x_aug_1, batch_y_aug_1 = augmenter(batch_x, batch_y)
                batch_x_aug_2, batch_y_aug_2 = augmenter(batch_x, batch_y)

                batch_x = batch_x.float().to(self.experiment.device)
                batch_y = batch_y.float().to(self.experiment.device)
                batch_x_mark = batch_x_mark.float().to(self.experiment.device)
                batch_y_mark = batch_y_mark.float().to(self.experiment.device)
                batch_x_aug_1 = batch_x_aug_1.float().to(self.experiment.device)
                batch_x_aug_2 = batch_x_aug_2.float().to(self.experiment.device)

                outputs_1 = self.experiment.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                outputs_2 = self.experiment.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                outputs_cat = torch.cat((outputs_1, outputs_2), dim=0)
                indices = torch.arange(0, outputs_1.shape[0], device=self.experiment.device)
                labels = torch.cat((indices, indices), dim=0)

                loss = criterion(outputs_cat, labels)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch{1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss = loss / self.experiment.args.accumulation_steps
                loss.backward()

                if (i + 1) % self.experiment.args.accumulation_steps == 0 or i == train_steps - 1:
                    model_optim.step()
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            epoch_losses.append(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, self.experiment.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch, self.experiment.args)

        print("Epoch Losses: ", epoch_losses)
        best_model_path = path + "/" + "checkpoint.pth"
        self.experiment.model.load_state_dict(torch.load(best_model_path))

        self.experiment.model.is_pretraining = False
    
        return self.experiment.model