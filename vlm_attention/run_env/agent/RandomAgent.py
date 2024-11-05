import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RandomAgent:
    """随机动作智能体，用于SC2环境测试"""
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, annotate_all_units: bool = True, grid_size: Tuple[int, int] = (10, 10),
                 use_self_attention: bool = False, use_rag: bool = False):
        """
        初始化随机动作智能体

        Args:
            action_space: 动作空间字典，包含 'attack' 和 'move' 两种动作类型
        """
        self.action_space = action_space
        self.step_count = 0

        # 获取单位数量上限(来自动作空间)
        self.num_units = self.action_space['attack'].spaces[0].n

        # 动作类型及其概率
        self.action_types = ['attack', 'move', 'none']
        self.action_probs = [0.3, 0.3, 0.4]  # 40%概率不执行动作

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, List[Tuple]]:
        """
        根据观察生成随机动作

        Args:
            observation: 包含游戏状态的字典

        Returns:
            action_dict: 包含attack和move动作列表的字典
        """
        self.step_count += 1

        # 初始化返回的动作字典
        action_dict = {
            'attack': [],
            'move': []
        }

        # 获取当前可用的单位信息
        available_units = [
            unit for unit in observation['unit_info']
            if unit['alliance'] == 1  # 只选择己方单位
        ]

        if not available_units:
            return action_dict

        # 随机选择动作类型
        action_type = np.random.choice(self.action_types, p=self.action_probs)

        if action_type == 'none':
            return action_dict

        # 根据动作类型生成随机动作
        if action_type == 'attack':
            # 随机选择攻击者和目标
            attacker_idx = np.random.randint(0, len(available_units))
            target_idx = np.random.randint(0, self.num_units)

            action_dict['attack'].append((
                available_units[attacker_idx]['simplified_tag'],
                target_idx
            ))

        elif action_type == 'move':
            # 随机选择移动单位和目标位置
            unit_idx = np.random.randint(0, len(available_units))
            move_type = np.random.randint(0, 3)  # 随机选择移动类型
            target_position = np.random.randint(0, 9, size=2)  # 随机选择目标位置

            action_dict['move'].append((
                move_type,
                available_units[unit_idx]['simplified_tag'],
                target_position
            ))

        # 记录日志
        logger.debug(f"Step {self.step_count}: Generated {action_type} action")

        return action_dict

    def reset(self):
        """重置智能体状态"""
        self.step_count = 0
        logger.info("Random agent reset")