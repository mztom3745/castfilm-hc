import logging
import os
import time
from loguru import logger


"""
设置日志在控制台的输出以及将不同日志保存在不同文件内
"""

# 设置日志文件记录路径
log_path = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)

# 配置logger的模块，用来记录系统生命周期相关记录
# 日志文件路径
log_path_main = os.path.join(log_path, '{time:YYYY-MM-DD}.log')
log_format = "{time:YYYY-MM-DD HH:mm:ss}  | {level} |  {message}"

def filter_system_monitor(record):
    """过滤掉监控日志和INFO级别日志（保留WARNING及以上级别的非监控日志）"""
    # 排除system_monitor上下文的日志
    if record["extra"].get("context") == "system_monitor":
        return False
    return True

def filter_system_monitor_and_info(record):
    """只保留INFO级别的非监控日志"""
    # 排除system_monitor上下文的日志
    if record["extra"].get("context") == "system_monitor":
        return False
    # 除error日志
    return record["level"].name != "ERROR"

logger.add(
    log_path_main,
    format=log_format,
    rotation='00:00',
    encoding='utf-8',
    enqueue=True,
    compression='zip',
    delay=0.1,
    filter=filter_system_monitor_and_info  # 使用相同过滤器
)

#错误日志
log_path_error = os.path.join(log_path, 'error.log')

logger.add(
    log_path_error,
    format=log_format,
    level="ERROR",             # 只记录ERROR及以上级别的日志
    rotation="10 MB",          # 当文件达到10MB时轮转
    retention="10 days",       # 保留10天的日志
    compression="zip",         # 压缩旧日志
    enqueue=True,              # 线程安全
    filter=filter_system_monitor  # 使用相同过滤器
)

# 配置logging模块
# 设置日志文件存储路径

# 获取根日志记录器
root_logger = logging.getLogger()
# 移除根日志记录器中处理器，防止重复操作
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)




# 设置控制台处理器
handler_console = logging.StreamHandler()
handler_console.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s  - %(message)s")
handler_console.setFormatter(formatter)


# 为各个日志记录器设置控制台处理器
root_logger.addHandler(handler_console)