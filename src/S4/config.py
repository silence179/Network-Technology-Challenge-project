MODE = "soft" # 模式选择，“soft”为mode b，其他是mode a, mode a为开发半成品，不推荐使用

csv_dir = "../S3/output/links" # csv文件夹路径，存储网络拓扑信息
rules_dir = "../S3/output/rules" # json文件夹路径，存储路由规则

sat_dir = "../S3/sat_trace/"
uav_csv = "../S3/uav_trace_full.csv"


from enum import Enum
class action(Enum):
    NOP = 1
    ADD = 2
    DEL = 3
    REPLACE = 4


import logging
import os
from logging.handlers import RotatingFileHandler

# ========================
# 全新 稳定版 日志模块
# ========================
class LogConfig:
    logger = None  # 单例模式，全局唯一 logger

    @staticmethod
    def get_logger():
        # 已经初始化过，直接返回，避免重复创建
        if LogConfig.logger is not None:
            return LogConfig.logger

        # 1. 创建 logger 单例
        logger = logging.getLogger("app_logger")
        logger.setLevel(logging.DEBUG)  # 最低接收级别
        logger.propagate = False  # 禁止向上传递，避免重复

        # 防止重复添加 handler
        if logger.handlers:
            logger.handlers.clear()

        # 2. 日志文件夹
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # ========================
        # 格式定义
        # ========================
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_formatter = logging.Formatter(
            "%(asctime)s | \033[%(color)sm%(levelname)-8s\033[0m | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # ========================
        # Handler 1：控制台彩色输出
        # ========================
        class ColorConsoleHandler(logging.StreamHandler):
            def format(self, record):
                colors = {
                    "DEBUG": "37",    # 灰色
                    "INFO": "32",     # 绿色
                    "WARNING": "33",  # 黄色
                    "ERROR": "31",    # 红色
                    "CRITICAL": "35"
                }
                record.color = colors.get(record.levelname, "37")
                return super().format(record)

        console_handler = ColorConsoleHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # ========================
        # Handler 2：INFO 级别文件（只存 INFO + 不包含 ERROR/WARNING）
        # ========================
        class InfoFilter(logging.Filter):
            def filter(self, record):
                return record.levelno == logging.INFO

        info_handler = RotatingFileHandler(
            os.path.join(log_dir, "info_runtime.log"),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        info_handler.setLevel(logging.INFO)
        info_handler.addFilter(InfoFilter())
        info_handler.setFormatter(file_formatter)
        logger.addHandler(info_handler)

        # ========================
        # Handler 3：WARNING 文件
        # ========================
        warn_handler = RotatingFileHandler(
            os.path.join(log_dir, "warning_runtime.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        warn_handler.setLevel(logging.WARNING)
        warn_handler.setFormatter(file_formatter)
        logger.addHandler(warn_handler)

        # ========================
        # Handler 4：ERROR 文件
        # ========================
        err_handler = RotatingFileHandler(
            os.path.join(log_dir, "error_runtime.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(file_formatter)
        logger.addHandler(err_handler)

        # ========================
        # Handler 5：DEBUG 文件
        # ========================
        debug_handler = RotatingFileHandler(
            os.path.join(log_dir, "debug_runtime.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        logger.addHandler(debug_handler)

        LogConfig.logger = logger
        return logger


# ========================
# 全局日志对象（你原来怎么用，现在还怎么用）
# ========================
log = LogConfig.get_logger()
