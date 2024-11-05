import json
import logging
import os
from typing import List, Tuple, Dict, Any
from datetime import datetime

import cv2
import numpy as np

from vlm_attention.run_env.utils import _annotate_units_on_image, _draw_grid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义颜色常量 (BGR 格式)
COLOR_SELF = (0, 255, 0)  # 绿色 (BGR)
COLOR_ENEMY = (0, 0, 255)  # 红色 (BGR)


class TestAgent:
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, annotate_all_units: bool = True, grid_size: Tuple[int, int] = (10, 10),
                 use_self_attention: bool = False, use_rag: bool = False):
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

        # logger.info(f"TestAgent initialized. Data will be saved in: {self.save_dir}")

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, List[Tuple]]:
        """处理观察并返回动作"""
        self.step_count += 1
        self.text_observation = observation["text"]
        # logger.info(f"Processing step {self.step_count}")
        # logger.info(f"Text observation: {self.text_observation}")

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

        # 返回空动作（测试用）
        return {
            'attack': [],
            'move': []
        }

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
                'position': [float(x) for x in unit['position']]
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

        frame_with_grid = _draw_grid(frame, screen_size, self.grid_size)

        cell_w, cell_h = w // self.grid_size[0], h // self.grid_size[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        font_color = (255, 255, 255)

        # 添加网格标签
        for i in range(self.grid_size[0]):
            x = i * cell_w + cell_w // 2
            cv2.putText(frame_with_grid, str(i), (x - 10, 30), font, font_scale, (0, 0, 0), font_thickness + 1)
            cv2.putText(frame_with_grid, str(i), (x - 10, 30), font, font_scale, font_color, font_thickness)

        for i in range(self.grid_size[1]):
            y = i * cell_h + cell_h // 2
            cv2.putText(frame_with_grid, str(i), (10, y + 10), font, font_scale, (0, 0, 0), font_thickness + 1)
            cv2.putText(frame_with_grid, str(i), (10, y + 10), font, font_scale, font_color, font_thickness)

        return frame_with_grid

    def _save_image(self, image: np.ndarray, directory: str, prefix: str) -> str:
        """保存图像到指定目录"""
        filename = os.path.join(directory, f"{prefix}_{self.step_count:04d}.png")
        cv2.imwrite(filename, image)
        return filename