# datasets/base_dataset.py
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import os
import numpy as np
from typing import Optional, Callable, Tuple, List, Any
from PIL import Image

class BaseDataset(Dataset, ABC):
    """自定义数据集的抽象基类"""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError(f"数据不存在，请设置 download=True 下载")
            
        self.data, self.targets = self._load_data()
    
    @abstractmethod
    def _check_exists(self) -> bool:
        """检查数据是否存在"""
        pass
    
    @abstractmethod
    def download(self) -> None:
        """下载数据（如果需要）"""
        pass
    
    @abstractmethod
    def _load_data(self) -> Tuple[List[Any], List[Any]]:
        """加载数据和标签"""
        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.data[idx], self.targets[idx]
        
        # 转换为PIL图像（如果需要）
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target