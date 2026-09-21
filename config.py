
import numpy as np


class GlobalConfig:
    """全局配置类"""
    _random_state = None
    _rng = None
    _is_initialized = False

    @classmethod
    def initialize(cls, random_state=42):
        """初始化全局配置"""
        if not cls._is_initialized:
            cls._random_state = random_state
            cls._rng = np.random.RandomState(random_state)
            np.random.seed(random_state)  # 同时设置numpy全局种子
            cls._is_initialized = True

    @classmethod
    def get_random_state(cls):
        """获取随机种子"""
        if not cls._is_initialized:
            cls.initialize()  # 使用默认值初始化
        return cls._random_state

    @classmethod
    def get_rng(cls):
        """获取随机数生成器"""
        if not cls._is_initialized:
            cls.initialize()
        return cls._rng

    @classmethod
    def reset(cls):
        """重置配置"""
        cls._random_state = None
        cls._rng = None
        cls._is_initialized = False


# 创建便捷函数
def get_rng():
    """获取全局随机数生成器"""
    return GlobalConfig.get_rng()


def set_random_state(seed):
    """设置全局随机种子"""
    GlobalConfig.initialize(seed)
