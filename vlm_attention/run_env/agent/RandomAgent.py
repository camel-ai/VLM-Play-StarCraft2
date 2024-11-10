import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RandomAgent:
    """随机动作智能体，用于SC2环境测试"""

    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, **kwargs):
        self.action_space = action_space
        self.step_count = 0

        # 获取单位数量上限
        self.num_units = self.action_space['attack'].spaces[0].n

        # 动作类型及其概率
        self.action_types = ['attack', 'grid_move', 'smac_move', 'none']
        self.action_probs = [0.2, 0.3, 0.3, 0.2]  # 60%概率执行移动

        # SMAC移动方向定义
        self.directions = range(4)  # UP(0), RIGHT(1), DOWN(2), LEFT(3)

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, List[Tuple]]:
        """根据观察生成随机动作"""
        self.step_count += 1

        # 初始化动作字典
        action_dict = {
            'attack': [],
            'move': []  # 包含grid_move和smac_move
        }

        # 获取当前可用的己方单位
        available_units = [
            unit for unit in observation['unit_info']
            if unit['alliance'] == 1
        ]

        if not available_units:
            return action_dict

        # 为每个己方单位生成动作
        for unit in available_units:
            # 随机决定是否为该单位生成命令(70%概率)
            if np.random.random() < 0.7:
                action_type = np.random.choice(self.action_types, p=self.action_probs)

                if action_type == 'attack':
                    # 随机选择一个目标
                    target_idx = np.random.randint(0, self.num_units)
                    action_dict['attack'].append((
                        unit['simplified_tag'],
                        target_idx
                    ))

                elif action_type == 'grid_move':
                    # Grid-based移动 (0-9范围内的坐标)
                    target_position = np.random.randint(0, 9, size=2)
                    action_dict['move'].append((
                        1,  # 移动类型1: grid-based移动
                        unit['simplified_tag'],
                        target_position.tolist()
                    ))

                elif action_type == 'smac_move':
                    # SMAC风格移动 (4个方向之一)
                    direction = np.random.choice(self.directions)
                    action_dict['move'].append((
                        2,  # 移动类型2: SMAC风格移动
                        unit['simplified_tag'],
                        [direction, 0]  # direction in [0,1,2,3], 第二个值填充0
                    ))

        # 记录日志
        logger.info(f"Step {self.step_count}: Generated actions:")
        logger.info(f"Attack actions: {action_dict['attack']}")
        logger.info(f"Move actions: {action_dict['move']}")

        return action_dict

    def reset(self):
        """重置智能体状态"""
        self.step_count = 0
        logger.info("Random agent reset")