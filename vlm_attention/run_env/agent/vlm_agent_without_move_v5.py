import json
import logging
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

from vlm_attention.env.config import COLORS, get_unit_name
from vlm_attention.knowledge_data.database.sc2_unit_database import SC2UnitDatabase
from vlm_attention.run_env.agent.agent_utils import (
    generate_important_units_prompt,
    generate_decision_prompt, parse_vlm_response, parse_vlm_decision,
    format_history_for_prompt, VLMPlanner,
    generate_unit_info_summary_prompt, generate_action_normalization_prompt, normalization_system_prompt
)
from vlm_attention.run_env.utils import _annotate_units_on_image, draw_grid_with_labels
from vlm_attention.utils.call_vlm import MultimodalChatbot, TextChatbot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VLMAgentWithoutMove:
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, grid_size: Tuple[int, int] = (10, 10),
                 use_self_attention: bool = False, use_rag: bool = False, history_length: int = 3,
                 replan_each_step: bool = False, use_proxy: bool = False):
        """
        初始化VLMAgentWithoutMove代理。
        :param action_space: 动作空间字典
        :param config_path: 配置文件路径
        :param save_dir: 保存目录
        :param draw_grid: 是否在截图上绘制网格
        :param annotate_units: 是否在截图上标注单位
        :param grid_size: 网格大小
        :param use_self_attention: 是否使用自注意力
        :param use_rag: 是否使用RAG
        """
        self.action_space = action_space
        self.important_units: List[int] = []
        self.text_observation: str = ""
        self.save_original_images = True
        self.history_length = history_length
        self.history: List[Dict[str, Any]] = []
        self.draw_grid = draw_grid
        self.annotate_units = annotate_units
        self.grid_size = grid_size
        self.step_count = 0
        self.use_self_attention = use_self_attention
        self.use_rag = use_rag
        self.use_proxy = use_proxy
        # 设置目录
        self.save_dir = save_dir
        self.original_images_dir = os.path.join(self.save_dir, "original_images")
        self.first_annotation_dir = os.path.join(self.save_dir, "first_annotation")
        self.second_annotation_dir = os.path.join(self.save_dir, "second_annotation")
        self.vlm_io_dir = os.path.join(self.save_dir, "vlm_io")
        self.unit_info_dir = os.path.join(self.save_dir, "unit_info")

        # 初始化数据库
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        yaml_path = os.path.join(project_root, 'vlm_attention', 'knowledge_data', 'database',
                                 'sc2_unit_data_index.yaml')

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found at {yaml_path}")

        self.db = SC2UnitDatabase(yaml_path)

        # 创建所需的所有目录
        for directory in [self.save_dir, self.original_images_dir, self.first_annotation_dir,
                          self.second_annotation_dir, self.vlm_io_dir, self.unit_info_dir]:
            os.makedirs(directory, exist_ok=True)

        # 初始化用于存储所有step数据的列表
        self.all_steps_data = []
        # 添加planner相关目录和初始化
        self.planner_dir = os.path.join(self.save_dir, "planner")
        os.makedirs(self.planner_dir, exist_ok=True)
        self.planner = VLMPlanner(self.planner_dir, replan_each_step)
        logger.info(f"VLMAgent initialized. Data will be saved in: {self.save_dir}")
        logger.info(f"Using YAML file: {yaml_path}")
        logger.info(f"Self-attention: {'ON' if self.use_self_attention else 'OFF'}")
        logger.info(f"RAG: {'ON' if self.use_rag else 'OFF'}")

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, List[Tuple]]:
        """获取动作决策

        主要修改:
        1. 使用_normalize_attack_action规范化攻击动作格式
        2. 移除所有move相关处理
        3. 返回规范化的动作字典
        """
        self.step_count += 1
        self.text_observation = observation["text"]
        logger.info(f"Processing step {self.step_count}")
        # logger.info(f"Text observation: {self.text_observation}")

        # 保存该step的unit info
        self._save_step_data(observation)

        # 保存原始图像
        if self.save_original_images:
            self._save_image(observation['image'], self.original_images_dir, "original")

        # 处理并保存第一次图像注释
        first_annotation_path = self._process_and_save_image(observation, self.first_annotation_dir, "first_annotation",
                                                             annotate_all=True)
        # 获取微操技能规划
        planned_skills = self.planner.plan(observation, first_annotation_path, use_proxy=self.use_proxy)
        logger.info(f"Planned micro skills: {planned_skills}")
        # 如果启用了自注意力机制，识别重要单位
        if self.use_self_attention:
            important_units_response = self._identify_important_units(
                observation,
                first_annotation_path,
                planned_skills
            )
            self.important_units = parse_vlm_response(important_units_response)
            logger.info(f"Identified important units: {self.important_units}")

            decision_image_path = self._process_and_save_image(
                observation,
                self.second_annotation_dir,
                "second_annotation",
                annotate_all=False
            )
        else:
            self.important_units = []
            important_units_response = "Self-attention is disabled."
            decision_image_path = first_annotation_path

        # RAG处理
        unit_summary = ""
        if self.use_rag:
            if self.use_self_attention:
                # 如果同时启用了自注意力，使用重要单位进行检索
                units_to_query = [unit for unit in observation['unit_info']
                                  if unit['alliance'] == 1 or unit['simplified_tag'] in self.important_units]
            else:
                # 如果没有启用自注意力，检索所有单位
                units_to_query = observation['unit_info']

            # 从数据库获取单位信息并生成总结 使用camel 框架
            unit_info = self._get_unit_info_from_database(units_to_query)
            """使用camel 框架"""
            unit_summary_system_prompt = "You are a StarCraft 2 expert focusing on micro-management."
            unit_summary_user_prompt = generate_unit_info_summary_prompt(unit_info)
            unit_summary_bot = TextChatbot(system_prompt=unit_summary_system_prompt, use_proxy=self.use_proxy)
            unit_summary = unit_summary_bot.query(unit_summary_user_prompt)
            unit_summary_bot.clear_history()
            """使用camel 框架结束"""
            self._save_vlm_io(unit_summary_user_prompt, unit_summary, "unit_summary")
            logger.info(f"单位信息总结：\n{unit_summary}")

        # 生成决策,使用camel 框架
        """使用camel 框架"""
        decision_system_prompt = generate_decision_prompt()
        decision_user_prompt = self._generate_decision_prompt(
            observation,
            unit_summary,
            important_units_response,
            planned_skills
        )
        raw_decision_bot = MultimodalChatbot(system_prompt=decision_system_prompt, use_proxy=self.use_proxy)
        raw_decision_response = raw_decision_bot.query(decision_user_prompt, image_path=decision_image_path)
        raw_decision_bot.clear_history()
        """使用camel 框架结束"""
        self._save_vlm_io(decision_user_prompt, raw_decision_response, "raw_decision", image_path=decision_image_path)

        # 解析原始决策并生成规范化提示
        raw_action = parse_vlm_decision(raw_decision_response)
        units_info_str = self.format_units_info_for_prompt(observation['unit_info'])
        normalization_user_prompt = generate_action_normalization_prompt(
            self.text_observation,
            units_info_str,
            raw_action
        )

        # 尝试规范化决策
        max_retries = 3
        normalized_action = {'attack': [], 'move': []}
        for attempt in range(max_retries):
            """使用camel 框架"""
            normalized_bot = TextChatbot(system_prompt=normalization_system_prompt(), use_proxy=self.use_proxy)
            normalized_action_response = normalized_bot.query(normalization_user_prompt)
            normalized_bot.clear_history()
            """使用camel 框架结束"""
            self._save_vlm_io(normalization_user_prompt, normalized_action_response,
                              f"normalized_decision_attempt_{attempt + 1}")

            normalized_action = parse_vlm_decision(normalized_action_response)
            if normalized_action['attack']:
                break
            elif attempt < max_retries - 1:
                logger.warning(f"Failed to parse actions on attempt {attempt + 1}. Retrying...")
            else:
                logger.error("Failed to parse actions after maximum retries. Returning empty actions.")

        # 添加无移动动作
        friendly_units = [unit['simplified_tag'] for unit in observation['unit_info'] if unit['alliance'] == 1]
        no_move_actions = [(0, unit, [0, 0]) for unit in friendly_units]
        normalized_action['move'] = no_move_actions

        # 更新历史记录
        self._update_history(
            self.important_units,
            {'attack': normalized_action['attack']},
            planned_skills
        )
        # 构建最终决策
        final_decision = {
            'attack': normalized_action['attack'],
            'move': normalized_action['move']
        }
        self._save_vlm_io("Final decision", json.dumps(final_decision), "final_decision")

        return final_decision

    def _normalize_attack_action(self, raw_attack_actions: List[Tuple], observation: Dict[str, Any]) -> List[Tuple]:
        """规范化攻击动作格式

        Args:
            raw_attack_actions: VLM生成的原始攻击动作列表 [(attacker_id, target_id),...]
            observation: 环境观察

        Returns:
            List[Tuple]: 规范化后的攻击动作列表
        """
        normalized_attacks = []

        for attacker_id, target_id in raw_attack_actions:
            # 确保ID为整数
            attacker_id = int(attacker_id)
            target_id = int(target_id)

            # 验证ID是否在合法范围内
            if 0 <= attacker_id < self.action_space['attack'][0].n and \
                    0 <= target_id < self.action_space['attack'][1].n:
                # 使用tuple而不是list,以匹配spaces.Tuple格式
                normalized_attacks.append((attacker_id, target_id))

        return normalized_attacks

    def _save_step_data(self, observation: Dict[str, Any]):
        """保存每个step的数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        step_data = {
            'step': self.step_count,
            'timestamp': timestamp,
            'text_observation': self.text_observation,
            'unit_info': []
        }

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

        self.all_steps_data.append(step_data)

        step_filename = os.path.join(self.unit_info_dir, f'step_{self.step_count:04d}.json')
        with open(step_filename, 'w', encoding='utf-8') as f:
            json.dump(step_data, f, indent=2, ensure_ascii=False)

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

        if self.draw_grid:
            frame = self._draw_grid_with_labels(frame)

        if self.annotate_units:
            units_to_annotate = []
            for unit in unit_info:
                color = COLORS['self_color'] if unit['alliance'] == 1 else COLORS['enemy_color']

                annotation_info = {
                    'tag_index': unit['simplified_tag'],
                    'label': unit['unit_name'],
                    'position': unit['position'],
                    'color': color
                }

                if annotate_all:
                    units_to_annotate.append(annotation_info)
                elif unit['alliance'] == 1 or unit['simplified_tag'] in self.important_units:
                    units_to_annotate.append(annotation_info)

            frame = _annotate_units_on_image(frame, units_to_annotate)

        filename = os.path.join(directory, f"{prefix}_{self.step_count:04d}.png")
        cv2.imwrite(filename, frame)
        return filename

    def _draw_grid_with_labels(self, frame: np.ndarray) -> np.ndarray:
        """在图像上绘制网格和标签"""
        h, w = frame.shape[:2]
        screen_size = (w, h)
        return draw_grid_with_labels(frame, screen_size, self.grid_size)

    def _identify_important_units(
            self,
            observation: Dict[str, Any],
            image_path: str,
            planned_skills: Dict[str, Any]
    ) -> str:
        """识别重要单位,与planned_skills紧密结合"""
        system_prompt = generate_important_units_prompt()
        units_info_str = self.format_units_info_for_prompt(observation['unit_info'])

        # 提取主要技能信息
        primary_skill = planned_skills.get('primary', {})
        skill_name = primary_skill.get('name', 'None')
        skill_desc = primary_skill.get('description', 'None')
        skill_steps = primary_skill.get('steps', [])

        user_input = f"""
            Analyze the current StarCraft II game state and identify important enemy units based on our planned micro skills:

            Primary Micro Skill Plan:
            Name: {skill_name}
            Description: {skill_desc}
            Implementation Steps:
            {chr(10).join(f"- {step}" for step in skill_steps)}

            Supporting Skills:
            {chr(10).join([
            f"- {skill.get('name', 'Unknown')}: {skill.get('description', 'No description')} "
            f"(Use when: {skill.get('condition', 'No condition specified')})"
            for skill in planned_skills.get('secondary', [])
        ])}

            Current Game State:
            {observation.get("text", "No text observation available.")}

            Units Information:
            {units_info_str}

            Previous Steps Context:
            {format_history_for_prompt(self.history, history_length=self.history_length)}

            Based on this information, identify enemy units that are:
            1. Most relevant to executing our {skill_name} strategy
            2. Could potentially disrupt our planned implementation steps
            3. Should be prioritized based on our micro skill requirements
            """

        important_unit_bot = MultimodalChatbot(system_prompt=system_prompt, use_proxy=self.use_proxy)
        important_units_response = important_unit_bot.query(user_input, image_path=image_path)
        important_unit_bot.clear_history()

        self._save_vlm_io(user_input, important_units_response, "important_units")
        return important_units_response

    def _generate_decision_prompt(self, observation: Dict[str, Any], unit_summary: str,
                                  important_units_response: str, planned_skills: Dict[str, Any]) -> str:
        """生成决策提示"""
        units_info_str = self.format_units_info_for_prompt(observation['unit_info'])

        # 获取主要技能信息
        primary_skill = planned_skills.get('primary', {})
        skill_name = primary_skill.get('name', 'None')
        skill_desc = primary_skill.get('description', 'None')
        skill_steps = primary_skill.get('steps', [])

        prompt = f"""
            CRITICAL INSTRUCTION:
            Your primary skill is {skill_name}. All actions must implement this skill according to these steps:
            {chr(10).join(skill_steps)}

            Current Game State Analysis:
            {self.text_observation}

            Available Units:
            {units_info_str}

            Micro-Management Plan:
            Primary Skill: {skill_name}
            Description: {skill_desc}
            Implementation Steps:
            {chr(10).join(skill_steps)}

            Supporting Skills:
            {chr(10).join([
            f"- {skill.get('name', 'Unknown')}: {skill.get('description', 'No description')} "
            f"(Use when: {skill.get('condition', 'No condition specified')})"
            for skill in planned_skills.get('secondary', [])
        ])}
            """

        if self.use_self_attention:
            prompt += f"""
                Priority Targets Analysis:
                {important_units_response}
                """

        if self.use_rag:
            prompt += f"""
                Unit Capabilities:
                {unit_summary}
                """

        prompt += f"""
            Recent History:
            {format_history_for_prompt(self.history, history_length=self.history_length)}

            FINAL INSTRUCTION:
            Generate attack actions that STRICTLY implement the {skill_name} skill.
            Each action's reasoning must explain how it contributes to executing this skill.
            """

        return prompt

    def _get_unit_info_from_database(self, units: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """从数据库获取单位信息"""
        unit_info = {}
        for unit in units:
            unit_type = unit['unit_type']
            base_name = get_unit_name(unit_type)

            info = self.db.get_unit_info(base_name)
            if info:
                unit_info[base_name] = info
            else:
                logger.warning(f"Can't find unit information for type {unit_type} ({base_name})")

        return unit_info

    def format_units_info_for_prompt(self, unit_info: List[Dict[str, Any]]) -> str:
        """为提示词格式化单位信息"""
        formatted_info = []
        for unit in unit_info:
            alliance = "Friendly" if unit['alliance'] == 1 else "Enemy"
            health_info = f"{unit['health']}/{unit['max_health']}"
            shield_info = f"{unit['shield']}/{unit['max_shield']}" if unit['max_shield'] > 0 else "No shields"
            energy_info = f"Energy: {unit['energy']}" if unit['energy'] > 0 else ""

            unit_desc = (
                f"{unit['unit_name']} ({alliance}, Tag: {unit['simplified_tag']})\n"
                f"Health: {health_info}, Shields: {shield_info}\n"
                f"{energy_info}\n"
                f"Position: [{unit['position'][0]:.1f}, {unit['position'][1]:.1f}]"
            )
            formatted_info.append(unit_desc)

        return "\n\n".join(formatted_info)

    def _save_vlm_io(self, prompt: str, response: str, prefix: str, image_path: Optional[str] = None) -> None:
        """保存VLM输入输出数据"""
        filename = os.path.join(self.vlm_io_dir, f"{prefix}_io_{self.step_count:04d}.json")
        data = {
            "prompt": prompt,
            "response": response,
            "step": self.step_count,
            "timestamp": datetime.now().isoformat()
        }

        if image_path:
            data["image_path"] = image_path

        try:
            if prefix in ["raw_decision", "normalized_decision"]:
                parsed_result = parse_vlm_decision(response)
                data["parsed_actions"] = parsed_result
                logger.debug(f"Parsed actions for {prefix}: {parsed_result}")
            elif prefix == "final_decision":
                data["parsed_actions"] = json.loads(response)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {prefix} I/O data to {filename}")
        except Exception as e:
            logger.error(f"Error saving {prefix} I/O data to {filename}: {e}", exc_info=True)

    def _save_image(self, image: np.ndarray, directory: str, prefix: str) -> str:
        """保存图像到指定目录"""
        filename = os.path.join(directory, f"{prefix}_{self.step_count:04d}.png")
        cv2.imwrite(filename, image)
        return filename

    def _update_history(
            self,
            important_units: List[int],
            action: Dict[str, List[Tuple]],
            planned_skills: Dict[str, Any]
    ):
        """更新历史记录"""
        self.history.append({
            "step": self.step_count,
            "important_units": important_units,
            "attack_actions": action['attack'],
            "planned_skills": planned_skills
        })
        if len(self.history) > self.history_length:
            self.history.pop(0)
