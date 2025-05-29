import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils.metric import Metric
from utils.logger import setup_logger

class BaseTrainer():
    def __init__(self, 
                 model, 
                 criterion, 
                 optimizer, 
                 device,
                 log_dir = './logs',
                 checkpoint_dir = './checkpoints',
                 scheduler = None, 
                 config = {
                     "epochs": 10,
                     "checkpoint_interval": 5,
                     "batch_log": False,
                     "use_tensorboard": True
                 }):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.config = config

        # 日志和检查点目录 
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tensorboard and logger
        self.batch_log = self.config.get("batch_log", False)
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.config.get("use_tensorboard", True) else None
        self.logger = setup_logger(name= self.model.__class__.__name__, 
                                   log_dir=self.log_dir,)

        # Training process
        self.current_epoch = 0
        self.best_metric = 0.

        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader, epochs):
        self.model.train()
        total_loss, total_acc = 0., 0.

        self.logger.info(f"Start training: {self.current_epoch}/{epochs} epoch")

        progress_bar = tqdm(train_loader, desc= f'Epoch {self.current_epoch+1}/{epochs} [Train]')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # 计算指标
            batch_acc = (output.argmax(dim=1) == target).float().mean().item()
            total_loss += loss.item()
            total_acc += batch_acc

            progress_bar.set_postfix(
                {"loss ": f"{total_loss/(batch_idx+1):4f}",
                 "acc": f"{total_acc/(batch_idx+1):.2%}",
                 "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                 }
            )

            # 记录 batch 日志
            if self.writer is not None and self.batch_log:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.current_epoch * len(train_loader) + batch_idx)
                self.writer.add_scalar('train/batch_acc', batch_acc, self.current_epoch * len(train_loader) + batch_idx)

        train_acc = Metric(total_acc, len(train_loader)).acc()
        train_loss = total_loss / len(train_loader)

        if self.writer is not None:
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, self.current_epoch)

        self.logger.info(f"Epoch {self.current_epoch+1} finished - train_loss: {train_loss:.4f}, train_acc: {train_acc:.2%}")

        return {
            "train_loss": train_loss,
            "train_acc": train_acc
        }
    
    def validate(self, val_loader, epochs):
        self.model.eval()   
        total_loss, total_acc = 0., 0.

        self.logger.info(f"Start validating: {self.current_epoch+1}/{epochs} epoch")

        process_bar = tqdm(val_loader, desc= f'Epoch {self.current_epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for data, target in process_bar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                total_acc+= (output.argmax(dim=1) == target).float().mean().item()
            
            val_acc = Metric(total_acc, len(val_loader)).acc()
            val_loss = total_loss / len(val_loader)
            if self.writer is not None:
                self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)

            self.logger.info(f"Epoch {self.current_epoch+1} finished - val_loss: {val_loss:.4f}, val_acc: {val_acc:.2%}")

            return{
                "val_loss": val_loss,
                "val_acc": val_acc
            }
    
    def train(self, train_loader, val_loader):
        """ 训练主函数

        Args:
            train_loader (DataLoader):  训练数据集 
            val_loader (DataLoader):    验证数据集
        """        
        self.logger.info(f"Start training, total epochs: {self.config['epochs']}")
        for epoch in range(self.config["epochs"]):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch(train_loader, self.config["epochs"])
            val_metrics = self.validate(val_loader, self.config["epochs"])

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else: 
                    self.scheduler.step()

            self._print_epoch_summary(train_metrics, val_metrics)
            self._handle_checkpoint(val_metrics)
        
        self.logger.info(f"Training finished - best val_acc: {self.best_metric:.2%}")
    def _print_epoch_summary(self, train_metrics, val_metrics):
        lr = self.optimizer.param_groups[0]['lr']
        print(
            f"Epoch {self.current_epoch+1}/{self.config['epochs']} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Train Acc: {train_metrics['train_acc']:.2%} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_acc']:.2%} | "
            f"LR: {lr:.2e}"
        )
    
    def _handle_checkpoint(self, val_metrics):
        """ 保存模型

        Args:
            val_metrics (dict):  验证集指标
        """        
        # 保存最佳模型
        if val_metrics["val_acc"] > self.best_metric:
            self.best_metric = val_metrics["val_acc"]
            self._save_checkpoint(is_best=True)

        # 定期保存
        if (self.current_epoch + 1) % self.config.get("checkpoint_interval", 5) == 0:
            self._save_checkpoint()
    def _save_checkpoint(self, is_best=False):
        """ 保存检查点

        Args:
            is_best (bool, optional): 是否为最佳. Defaults to False.
        """        
        checkpoint={
            "timestamp": datetime.now().isoformat(),
            "epoch": self.current_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "schedule_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_metric": self.best_metric,
            "config": self.config
        }

        file_name = f"checkpoint_epoch{self.current_epoch+1}.pth"
        if is_best: 
            file_name = "best_model.pth"
            self.logger.info(f"Saved best model, accuracy: {self.best_metric:.2%}")
        try:
            torch.save(checkpoint, self.checkpoint_dir/file_name)
            checkpoint_size = (self.checkpoint_dir/file_name).stat().st_size / 1024 ** 2

            if is_best:
                self.logger.info(f"Saved best model (size: {checkpoint_size:.2f} MB), accuracy: {self.best_metric:.2%}")
            else:
                self.logger.info(f"Saved checkpoint (size: {checkpoint_size:.2f} MB): {self.checkpoint_dir/file_name}")
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")

    def load_checkpoint(self, checkpoint_path):
        try:
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            if self.scheduler and checkpoint["schedule_state"]:
                self.scheduler.load_state_dict(checkpoint["schedule_state"])

            self.best_metric = checkpoint["best_metric"]
            self.current_epoch = checkpoint["epoch"]

            self.logger.info(f"Loaded checkpoint - epoch: {self.current_epoch+1}, best_metric: {self.best_metric:.2%}")
            return checkpoint["epoch"]
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise