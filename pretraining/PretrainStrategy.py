from utils.aug_strategy import BasicAUG

class PretrainStrategy(object):
    
    def __init__(self, exp):
        self.experiment = exp

        self.aug_strategy_dict = {
            "BasicAUG": BasicAUG,
        }

        self.augmenter = self._build_aug_strategy().to(self.experiment.device)

    def _build_aug_strategy(self):
        return self.aug_strategy_dict[self.experiment.args.aug_strategy](self.experiment.args)

    def _select_criterion(self):
        pass

    def pretrain(self):
        pass