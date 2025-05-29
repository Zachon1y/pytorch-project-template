import logging
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_dir, level=logging.INFO):
    """配置并返回一个日志记录器
    Args:
        name (str): 日志记录器的名称，通常使用模块名或类名
        log_dir (str/Path): 日志文件保存的目录路径
        level (int): 日志级别，默认使用INFO级别

    Returns:
        logging.Logger: 配置好的日志记录器对象
    """    
    logger = logging.getLogger(name) # 获取指定名称的日志记录器
    logger.setLevel(level)  # 获取指定名称的日志记录器

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 控制台处理器：将日志输出到终端
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器（可选）：将日志写入文件
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True) # 创建日志目录
            safe_name = name.replace('.', '_')
            file_path = log_dir/ f'{safe_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger