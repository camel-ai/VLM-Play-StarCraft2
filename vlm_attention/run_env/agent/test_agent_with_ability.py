import json
import logging
import os
from typing import List, Tuple, Dict, Any
from datetime import datetime

import cv2
import numpy as np

from vlm_attention.run_env.utils import _annotate_units_on_image, draw_grid_with_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义颜色常量 (BGR 格式)
COLOR_SELF = (0, 255, 0)  # 绿色 (BGR)
COLOR_ENEMY = (0, 0, 255)  # 红色 (BGR)


"""

test agent for the environment with ability support

It`s action space is :

move, attack, ability

"""
class TestAgent_With_Ability:
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, annotate_all_units: bool = True, grid_size: Tuple[int, int] = (10, 10),
                 use_self_attention: bool = False, use_rag: bool = False, use_proxy: bool = False,
                 model_name: str = "qwen"):
        self.action_space = action_space

        self.text_observation: str = ""
        self.save_original_images = True
        self.history: List[Dict[str, Any]] = []
        self.draw_grid = draw_grid
        self.annotate_units = annotate_units
        self.annotate_all_units = annotate_all_units
        self.grid_size = grid_size
        self.step_count = 0

        # 加载配置文件
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # 设置目录
        self.save_dir = save_dir
        self.original_images_dir = os.path.join(self.save_dir, "original_images")
        self.first_annotation_dir = os.path.join(self.save_dir, "first_annotation")
        self.unit_info_dir = os.path.join(self.save_dir, "unit_info")

        # 创建所需的所有目录
        for directory in [self.save_dir, self.original_images_dir,
                          self.first_annotation_dir, self.unit_info_dir]:
            os.makedirs(directory, exist_ok=True)

        # 初始化用于存储所有step数据的列表
        self.all_steps_data = []
        # 获取单位数量上限
        self.num_units = self.action_space['attack'].spaces[0].n

        # 更新动作类型及其概率
        self.action_types = ['attack', 'grid_move', 'smac_move', 'ability', 'none']
        self.action_probs = [0.0, 0.0, 0.0, 0.9, 0.1]  # 调整概率分布

        # SMAC移动方向定义
        self.directions = range(4)  # UP(0), RIGHT(1), DOWN(2), LEFT(3)

        # logger.info(f"TestAgent initialized. Data will be saved in: {self.save_dir}")

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, List[Tuple]]:
        """处理观察并返回动作"""
        self.step_count += 1
        self.text_observation = observation["text"]
        
        # 添加详细的观测信息打印
        print("\n=== Observation Details ===")
        print("\nUnit Information:")
        for unit in observation['unit_info']:
            print(f"\nUnit: {unit['unit_name']} (Tag: {unit['simplified_tag']}, Alliance: {unit['alliance']})")
            print(f"Health: {unit['health']}/{unit['max_health']}")
            print(f"Position: {unit['position']}")
            
            # 特别关注技能信息
            if 'abilities' in unit:
                print("Abilities:")
                for idx, ability in enumerate(unit['abilities']):
                    print(f"  [{idx}] {ability}")
            else:
                print("No abilities found")
                
            if 'active_abilities' in unit:
                print("Active abilities:", unit['active_abilities'])
                
            if 'ability_cooldowns' in unit:
                print("Ability cooldowns:", unit['ability_cooldowns'])
        
        print("\n=== End of Observation Details ===\n")

        # 保存原始图像
        if self.save_original_images:
            self._save_image(observation['image'], self.original_images_dir, "original")

        # 处理并保存带注释的图像
        first_annotation_path = self._process_and_save_image(
            observation,
            self.first_annotation_dir,
            "first_annotation",
            annotate_all=True
        )

        # 保存该step的unit info
        self._save_step_data(observation)

        # 初始化动作字典
        action_dict = {
            'attack': [],
            'move': [],  # 包含grid_move和smac_move
            'ability': []  # 新增技能命令列表
        }

        # 获取当前可用的己方单位和敌方单位
        available_units = [
            unit for unit in observation['unit_info']
            if unit['alliance'] == 1
        ]
        enemy_units = [
            unit for unit in observation['unit_info']
            if unit['alliance'] == 4
        ]

        if not available_units:
            return action_dict

        # 为每个己方单位独立决定是否生成动作，以及生成什么类型的动作
        for unit in available_units:
            if 'simplified_tag' not in unit:
                continue

            # 首先决定是否为这个单位生成动作（50%概率）
            if np.random.random() < 0.5:
                # 然后从可用的动作类型中随机选择一个
                action_type = np.random.choice(self.action_types, p=self.action_probs)
                
                if action_type == 'attack' and enemy_units:
                    # 生成攻击动作...
                    target = np.random.choice(enemy_units)
                    action_dict['attack'].append((
                        unit['simplified_tag'],
                        target['simplified_tag']
                    ))
                    
                elif action_type == 'grid_move':
                    # 生成网格移动动作...
                    x = np.random.randint(0, 10)
                    y = np.random.randint(0, 10)
                    action_dict['move'].append((1, unit['simplified_tag'], [x, y]))
                    
                elif action_type == 'smac_move':
                    # 生成SMAC移动动作...
                    direction = np.random.choice(self.directions)
                    action_dict['move'].append((2, unit['simplified_tag'], [direction, 0]))
                    
                elif action_type == 'ability' and 'abilities' in unit and unit['abilities']:
                    # 生成技能动作...
                    ability = np.random.choice(unit['abilities'])
                    target_type = ability.get('target_type', 0)
                    target_info = {'target_type': target_type}
                    
                    # 根据技能类型设置目标信息
                    if target_type == 1:  # POINT
                        target_info['position'] = [
                            np.random.randint(0, 10),
                            np.random.randint(0, 10)
                        ]
                        target_info['target_unit'] = 0
                    elif target_type == 2:  # UNIT
                        if 'LOAD' in ability.get('name', '').upper():
                            # 选择友方单位
                            if available_units:
                                target_unit = np.random.choice(available_units)
                                target_info['target_unit'] = target_unit['simplified_tag']
                            else:
                                continue
                        else:
                            # 选择敌方单位
                            if enemy_units:
                                target_unit = np.random.choice(enemy_units)
                                target_info['target_unit'] = target_unit['simplified_tag']
                            else:
                                continue
                        target_info['position'] = [0, 0]
                    else:  # QUICK or AUTO
                        target_info['position'] = [0, 0]
                        target_info['target_unit'] = 0
                    
                    ability_index = unit['abilities'].index(ability)
                    action_dict['ability'].append((
                        unit['simplified_tag'],
                        ability_index,
                        target_info
                    ))

        # 记录生成的动作
        logger.info(f"\nStep {self.step_count} Generated Actions:")
        logger.info(f"Attack actions: {action_dict['attack']}")
        logger.info(f"Move actions: {action_dict['move']}")
        logger.info(f"Ability actions: {action_dict['ability']}")

        return action_dict

    def _save_step_data(self, observation: Dict[str, Any]):
        """保存每个step的数据"""
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 准备该step的数据
        step_data = {
            'step': self.step_count,
            'timestamp': timestamp,
            'text_observation': self.text_observation,
            'unit_info': []
        }

        # 处理每个单位的信息
        for unit in observation['unit_info']:
            unit_data = {
                'simplified_tag': unit['simplified_tag'],
                'original_tag': unit['original_tag'],
                'alliance': unit['alliance'],
                'unit_type': unit['unit_type'],
                'unit_name': unit['unit_name'],
                'health': float(unit['health']),
                'max_health': float(unit['max_health']),
                'shield': float(unit['shield']),
                'max_shield': float(unit['max_shield']),
                'energy': float(unit['energy']),
                'position': [float(x) for x in unit['position']],
                'abilities': unit.get('abilities', [])
            }
            step_data['unit_info'].append(unit_data)

        # 将该step的数据添加到总数据列表中
        self.all_steps_data.append(step_data)

        # 每个step都保存一个独立的JSON文件
        step_filename = os.path.join(self.unit_info_dir, f'step_{self.step_count:04d}.json')
        with open(step_filename, 'w', encoding='utf-8') as f:
            json.dump(step_data, f, indent=2, ensure_ascii=False)

        # 同时更新总的数据文件
        complete_data = {
            'total_steps': self.step_count,
            'steps': self.all_steps_data
        }

        complete_filename = os.path.join(self.save_dir, 'complete_data.json')
        with open(complete_filename, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)

    def _process_and_save_image(self, observation: Dict[str, Any], directory: str, prefix: str,
                                annotate_all: bool) -> str:
        """处理和保存图像"""
        frame = observation['image'].copy()
        unit_info = observation['unit_info']
        # logger.debug("Processing unit_info for image annotation")

        if self.draw_grid:
            # logger.debug("Drawing grid on frame")
            frame = self._draw_grid_with_labels(frame)

        if self.annotate_units:
            # 转换单位信息为标注格式
            units_to_annotate = []
            for unit in unit_info:
                # 设置颜色（根据alliance）
                color = COLOR_SELF if unit['alliance'] == 1 else COLOR_ENEMY

                # 构建标注信息
                annotation_info = {
                    'tag_index': unit['simplified_tag'],
                    'position': unit['position'],
                    'color': color
                }
                units_to_annotate.append(annotation_info)

            # 使用utils中的函数进行标注
            frame = _annotate_units_on_image(
                frame,
                units_to_annotate
            )

        filename = os.path.join(directory, f"{prefix}_{self.step_count:04d}.png")
        cv2.imwrite(filename, frame)
        # logger.debug(f"{prefix.capitalize()} saved: {filename}")
        return filename

    def _draw_grid_with_labels(self, frame: np.ndarray) -> np.ndarray:
        """在图像上绘制网格和标签"""
        h, w = frame.shape[:2]
        screen_size = (w, h)
        return draw_grid_with_labels(frame, screen_size, self.grid_size)

    def _save_image(self, image: np.ndarray, directory: str, prefix: str) -> str:
        """保存图像到指定目录"""
        filename = os.path.join(directory, f"{prefix}_{self.step_count:04d}.png")
        cv2.imwrite(filename, image)
        return filename
