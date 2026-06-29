import numpy as np
import logging
import os
import logging

logs = set()

import logging
import os

def init_log(name='global', level=logging.INFO, log_dir='logs', filename="train.log"):
    """
    初始化日志系统
    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志文件目录
        filename: 日志文件名，例如 'train.log'
    """
    os.makedirs(log_dir, exist_ok=True)

    # 如果未指定文件名，自动用 name 命名
    if filename is None:
        filename = f"{name}.log"
    
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if not logger.handlers:
        # 文件输出
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 格式
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count
