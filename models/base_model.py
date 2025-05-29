import torch
import torch.nn as nn
import numpy as np
import os
from abc import abstractmethod
from datetime import datetime

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, config = None):
        """ 
        Constructor for the BaseModel class

        Args:
            config (str, optional): model config dict. Defaults to None.
        """        
        super().__init__()
        self.config = config or {}
        # self.logger = logging.getLogger(self.__class__.__name__) # 创建日志记录器

    @abstractmethod
    def forward(self, x):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def init_weights(self, method = 'kaiming'):
        """
            Initialize model weights: kaiming, xavier, normal
            :param method: init method
            :return: None
        """      

        for m in self.modules():
            if isinstance (m, nn.Conv2d):
                if method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean = 0, std = 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance (m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance (m, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif method == 'normal':
                    nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # self.logger.info("Weights initialized")


    def save(self, save_dir, filename = 'model.pth', optimizer = None):
        """ Save model to a given directory
        Args:
            save_dir (str):  directory to save model
            filename (str, optional): Defaults to 'model.pth'.
            optimizer (torch.optim, optional): Defaults to None.
        """
        state_dict = {
            'timestamp': datetime.now().isoformat(),
            'model': self.state_dict(),
            'config': self.config,
        }

        if optimizer is not None:
            state_dict['optimizer'] = optimizer.state_dict()

        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state_dict, save_path)

        # self.logger.info(f"Model saved to {save_path}")

    def load(self, 
            load_path,
            device = None,
            load_optimizer = False,
            optimizer = None):
        """ Load model from file

        Args:
            load_dir (str): Model file path
            device (torch.device): Device to load model to. Defaults to None.
            load_optimizer (bool, optional): Load optimizer. Defaults to False.
            optimizer (Optional[torch.optim.Optimizer], optional): Optimizer to load. Defaults to None.
        
        Returns:
            []: Model loaded from file
        """
        if not os.path.exists(load_path):
            raise ValueError(f"Model file{load_path} does not exist")       
        
        checkpoint = torch.load(load_path, map_location=device)
        self.load_state_dict(checkpoint['model'])
        result = {
            'model_loaded': True,
        }

        if load_optimizer and 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            result['optimizer_loaded'] = True

        return result

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)