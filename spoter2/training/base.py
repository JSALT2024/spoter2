import torch


class BaseTrainer:
    def __init__(
            self,
            config: dict,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.epochs = self.config["epochs"]
        self.epoch = 0
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()

