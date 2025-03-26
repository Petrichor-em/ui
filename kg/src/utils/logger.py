import logging
import os
from datetime import datetime

# 获取项目根目录路径
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_logger = None

def setup_logger(log_file_path=None):
    """
    设置logger，确保只创建一次logger实例
    
    Args:
        log_file_path (str): 日志文件存储路径，默认为项目根目录下的'logs'目录
    
    Returns:
        logging.Logger: 配置好的logger对象
    """
    global _logger
    if _logger is not None:
        return _logger

    if log_file_path is None:
        log_file_path = os.path.join(project_dir, 'logs')
    
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    
    log_file_name = f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_file = os.path.join(log_file_path, log_file_name)
    
    logger = logging.getLogger(log_file_name)  # 使用唯一的logger名称
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    _logger = logger
    return logger

# 创建全局logger实例
logger = setup_logger()

# 使用示例
if __name__ == '__main__':
    logger.debug('这是一条调试信息')
    logger.info('这是一条普通信息')
    logger.warning('这是一条警告信息')
    logger.error('这是一条错误信息')
    logger.critical('这是一条严重错误信息')
