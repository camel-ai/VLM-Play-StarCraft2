import json
import logging
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

from vlm_attention.env.config import COLORS, get_unit_name
from vlm_attention.knowledge_data.database.sc2_unit_database import SC2UnitDatabase
from vlm_attention.run_env.agent.agent_ability_utils import (
    generate_important_units_prompt,
    generate_decision_prompt, parse_vlm_response, parse_vlm_decision,
    format_history_for_prompt, VLMPlanner,
    generate_unit_info_summary_prompt, generate_action_normalization_prompt, normalization_system_prompt,
    normalize_important_units
)
from vlm_attention.run_env.agent.role_assignment import RoleAssignment
from vlm_attention.run_env.utils import _annotate_units_on_image, draw_grid_with_labels
from vlm_attention.utils.call_vlm import MultimodalChatbot, TextChatbot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
This is an agent with ability support.

It`s action space is :
move, attack, ability

"""
class VLMAgentWithAbility:
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, grid_size: Tuple[int, int] = (10, 10),
                 use_self_attention: bool = False, use_rag: bool = False, history_length: int = 3,
                 replan_each_step: bool = False, use_proxy: bool = False, move_type: str = 'grid',
                 rgb_screen_width: int = 1920, rgb_screen_height: int = 1080):
        """
        初始化VLMAgentWithAbility代理。
        :param action_space: 动作空间字典
        :param config_path: 配置文件路径
        :param save_dir: 保存目录
        :param draw_grid: 是否在截图上绘制网格
        :param annotate_units: 是否在截图上标注单位
        :param grid_size: 网格大小
        :param use_self_attention: 是否使用自注意力
        :param use_rag: 是否使用RAG
        :param move_type: 移动方式 ('grid' 或 'smac')
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
        self.image_size = (rgb_screen_width, rgb_screen_height)
        # 设置目录结构
        self.save_dir = save_dir
        self.original_images_dir = os.path.join(self.save_dir, "original_images")
        self.first_annotation_dir = os.path.join(self.save_dir, "first_annotation")
        self.second_annotation_dir = os.path.join(self.save_dir, "second_annotation")
        self.vlm_io_dir = os.path.join(self.save_dir, "vlm_io")
        self.unit_info_dir = os.path.join(self.save_dir, "unit_info")

        # 新增: Planner相关目录和初始化
        self.planner_dir = os.path.join(self.save_dir, "planner")
        os.makedirs(self.planner_dir, exist_ok=True)
        self.planner = VLMPlanner(self.planner_dir, replan_each_step)
        self.role_assignment = RoleAssignment(config_path)
        self.current_assignments = {}
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

        # 添加移动方式设置
        self.move_type = move_type
        if move_type not in ['grid', 'smac']:
            raise ValueError("move_type must be either 'grid' or 'smac'")

        # 根据移动方式设置相应的move_type_id
        self.move_type_id = 1 if move_type == 'grid' else 2

        # 新增: 技能相关的目录
        self.ability_dir = os.path.join(self.save_dir, "ability")
        os.makedirs(self.ability_dir, exist_ok=True)

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, List[Tuple]]:
        """获取动作决策,整合了技能规划和ability使用"""
        self.step_count += 1
        self.text_observation = observation["text"]
        logger.info(f"Processing step {self.step_count}")

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
                units_to_query = [unit for unit in observation['unit_info']
                                  if unit['alliance'] == 1 or unit['simplified_tag'] in self.important_units]
            else:
                units_to_query = observation['unit_info']

            unit_info = self._get_unit_info_from_database(units_to_query)
            unit_summary_system_prompt = "You are a StarCraft 2 expert focusing on micro-management and ability usage."
            unit_summary_user_prompt = generate_unit_info_summary_prompt(unit_info)
            unit_summary_bot = TextChatbot(system_prompt=unit_summary_system_prompt, use_proxy=self.use_proxy)
            unit_summary = unit_summary_bot.query(unit_summary_user_prompt)
            unit_summary_bot.clear_history()
            self._save_vlm_io(unit_summary_user_prompt, unit_summary, "unit_summary")

        # 添加调试日志
        for unit in observation['unit_info']:
            if unit['alliance'] == 1:  # 只检查友方单位
                logger.info(f"Unit {unit['simplified_tag']} ({unit['unit_name']}) abilities:")
                if 'abilities' in unit:
                    for ability in unit['abilities']:
                        logger.info(f"  - {ability}")
                else:
                    logger.warning(f"No abilities found for unit {unit['simplified_tag']}")

        # 生成决策
        decision_system_prompt = generate_decision_prompt()
        decision_user_prompt = self._generate_decision_prompt(
            observation,
            unit_summary,
            important_units_response,
            planned_skills,
            self.current_assignments
        )
        raw_decision_bot = MultimodalChatbot(system_prompt=decision_system_prompt, use_proxy=self.use_proxy)
        raw_decision_response = raw_decision_bot.query(decision_user_prompt, image_path=decision_image_path)
        raw_decision_bot.clear_history()
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
        normalized_action = {'attack': [], 'move': [], 'ability': []}  # 添加ability列表
        for attempt in range(max_retries):
            normalized_bot = TextChatbot(system_prompt=normalization_system_prompt(), use_proxy=self.use_proxy)
            normalized_action_response = normalized_bot.query(normalization_user_prompt)
            normalized_bot.clear_history()
            self._save_vlm_io(normalization_user_prompt, normalized_action_response,
                              f"normalized_decision_attempt_{attempt + 1}")

            # 使用move_type解析动作
            normalized_action = parse_vlm_decision(normalized_action_response)
            if normalized_action['attack'] or normalized_action['move'] or normalized_action['ability']:
                break
            elif attempt < max_retries - 1:
                logger.warning(f"Failed to parse actions on attempt {attempt + 1}. Retrying...")
            else:
                logger.error("Failed to parse actions after maximum retries. Returning empty actions.")

        # 更新历史记录
        self._update_history(
            self.important_units,
            normalized_action,
            planned_skills
        )

        # 获取有效的单位tag
        valid_friendly_tags = [unit['simplified_tag'] for unit in observation['unit_info']
                               if unit['alliance'] == 1]
        valid_enemy_tags = [unit['simplified_tag'] for unit in observation['unit_info']
                            if unit['alliance'] != 1]

        logger.info(f"Valid friendly tags: {valid_friendly_tags}")
        logger.info(f"Valid enemy tags: {valid_enemy_tags}")

        # 验证攻击动作
        if normalized_action['attack']:
            attack_actions = []
            for attacker_tag, target_tag in normalized_action['attack']:
                if (isinstance(attacker_tag, int) and isinstance(target_tag, int) and
                        attacker_tag in valid_friendly_tags and target_tag in valid_enemy_tags):
                    attack_actions.append((attacker_tag, target_tag))
                else:
                    logger.warning(f"Invalid attack action: attacker={attacker_tag}, target={target_tag}")
            normalized_action['attack'] = attack_actions

        # 验证移动动作
        if normalized_action['move']:
            move_actions = []
            for move_action in normalized_action['move']:
                if len(move_action) == 2:
                    unit_tag, target = move_action
                    if unit_tag not in valid_friendly_tags:
                        logger.warning(f"Invalid unit tag for move: {unit_tag}")
                        continue

                    if isinstance(target, list):
                        if self.move_type == 'grid' and len(target) == 2:
                            x, y = target
                            if 0 <= x <= 9 and 0 <= y <= 9:
                                move_actions.append((1, unit_tag, [x, y]))
                        elif self.move_type == 'smac' and len(target) == 2:
                            direction = target[0]
                            if 0 <= direction <= 3:
                                move_actions.append((2, unit_tag, [direction, 0]))
            normalized_action['move'] = move_actions

        # 验证技能动作
        if normalized_action['ability']:
            ability_actions = []
            for ability_action in normalized_action['ability']:
                unit_tag, ability_index, target_info = ability_action
                unit = next((u for u in observation['unit_info'] if u['simplified_tag'] == unit_tag), None)
                
                if unit and 'abilities' in unit:
                    # 打印调试信息
                    logger.info(f"Unit {unit_tag} abilities: {unit['abilities']}")
                    
                    # 获取单位的可用技能
                    available_abilities = unit.get('abilities', [])
                    if available_abilities:
                        # 确保ability_index在有效范围内
                        if 0 <= ability_index < len(available_abilities):
                            ability = available_abilities[ability_index]
                            
                            # 验证目标信息
                            target_type = ability.get('target_type')
                            valid_target = self._validate_ability_target(
                                target_type, 
                                target_info, 
                                valid_friendly_tags,
                                valid_enemy_tags
                            )
                            
                            if valid_target:
                                ability_actions.append((unit_tag, ability_index, target_info))
                                logger.info(f"Added valid ability action for unit {unit_tag}: {ability}")
                            else:
                                logger.warning(f"Invalid ability target for unit {unit_tag}: {target_info}")
                        else:
                            logger.warning(f"Ability index {ability_index} out of range for unit {unit_tag}. "
                                         f"Available abilities: {len(available_abilities)}")
                    else:
                        logger.warning(f"No abilities available for unit {unit_tag}")
                else:
                    logger.warning(f"Unit {unit_tag} not found or has no abilities defined")
                
            normalized_action['ability'] = ability_actions

        logger.info(f"Final normalized actions: {normalized_action}")

        # 在合适的时机更新角色分配
        if self._should_update_assignments(observation):
            new_assignments = self.role_assignment.initial_assignment(observation)
            tasks = list(new_assignments.values())
            task_similarities = self.role_assignment.contrastive_learning(tasks)
            battle_outcome = self._evaluate_battle_outcome(observation)
            adjusted_assignments = self.role_assignment.reflect_and_adjust(
                new_assignments, battle_outcome
            )
            self.current_assignments = adjusted_assignments

        return normalized_action

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
                'position': [float(x) for x in unit['position']],
                'grid_position': unit['grid_position']
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

        user_input = f"""Analyze the current StarCraft II game state and identify important enemy units.

CURRENT SITUATION:
Primary Micro Skill: {skill_name}
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

REQUIRED FORMAT EXAMPLE:
## Important Units ##
Unit: Zealot
Tag: 7
Reason: Highest shield value (50/50) and positioned closest to our units at [0,1], immediate threat to our Reapers

## Important Units ##
Unit: Zealot
Tag: 12
Reason: Full shields and blocking key escape route at [1,1], could trap our units if not dealt with

YOUR TASK:
Identify at least 2-3 important enemy units that:
1. Are most relevant to executing our {skill_name} strategy
2. Could potentially disrupt our planned implementation steps
3. Should be prioritized based on our micro skill requirements

CRITICAL RULES:
- You MUST use the exact format shown in the example above
- You MUST identify at least 2-3 units
- You MUST include exact tag numbers
- You MUST provide detailed tactical reasons
- You MUST focus on units with tactical advantages (high health/shields, threatening positions)
"""

        # 获取初始响应
        important_unit_bot = MultimodalChatbot(system_prompt=system_prompt, use_proxy=self.use_proxy)
        raw_response = important_unit_bot.query(user_input, image_path=image_path)
        important_unit_bot.clear_history()
        
        # 保存原始响应
        self._save_vlm_io(user_input, raw_response, "important_units_raw")

        # 规范化响应
        normalized_response = normalize_important_units(raw_response, observation, self.use_proxy)
        
        # 保存规范化后的响应
        self._save_vlm_io(user_input, normalized_response, "important_units_normalized")

        return normalized_response

    def _generate_decision_prompt(self, observation: Dict[str, Any], unit_summary: str,
                                  important_units_response: str, planned_skills: Dict[str, Any],
                                  current_assignments: Dict = None) -> str:
        """生成决策提示,整合了技能规划信息"""
        # 添加技能信息的日志
        for unit in observation['unit_info']:
            if 'abilities' in unit:
                logger.info(f"Unit {unit['simplified_tag']} ({unit['unit_name']}) abilities: {unit['abilities']}")
        
        # 提取主要技能信息
        primary_skill = planned_skills.get('primary', {})
        skill_name = primary_skill.get('name', 'None')
        skill_desc = primary_skill.get('description', 'None')
        skill_steps = primary_skill.get('steps', [])
        skill_steps_str = "\n".join(skill_steps)

        # 直接使用包含grid_position的单位信息
        units_info_str = self.format_units_info_for_prompt(observation['unit_info'])

        # 根据移动方式生成不同的移动系统说明
        if self.move_type == 'grid':
            movement_system = f"""
            MOVEMENT SYSTEM (CRITICAL):
            1. Use ONLY grid coordinates (0-9), NOT pixel or game coordinates
            2. Map is divided into a 10x10 grid:
               - [0,0] is top-left corner
               - [9,9] is bottom-right
               - Each unit's position is given in grid coordinates
            3. Movement Format:
               Move: [Unit Tag] -> [x, y]
               where x and y are integers between 0 and 9
            4. EXAMPLES:
               CORRECT: Move: 1 -> [5, 3]
               INCORRECT: Move: 1 -> [855, 518]
               INCORRECT: Move: 1 -> [x + 10, y + sqrt(2)]
            """
        else:  # SMAC movement
            movement_system = f"""
            MOVEMENT SYSTEM (CRITICAL):
            1. Use ONLY cardinal directions (0-3):
               0: UP (move up)
               1: RIGHT (move right)
               2: DOWN (move down)
               3: LEFT (move left)
            2. Movement Format:
               Move: [Unit Tag] -> [direction]
            3. EXAMPLES:
               CORRECT: Move: 1 -> [0]  (move up)
               CORRECT: Move: 2 -> [1]  (move right)
               INCORRECT: Move: 1 -> [5, 3]
               INCORRECT: Move: 2 -> [4]
            """

        # 添加可用技能信息
        available_abilities_str = "\nAvailable Unit Abilities:\n"
        for unit in observation['unit_info']:
            if unit['alliance'] == 1 and 'abilities' in unit:  # 只显示友方单位的技能
                abilities_info = []
                for idx, ability in enumerate(unit['abilities']):
                    target_type_str = {
                        0: "QUICK (no target needed)",
                        1: "POINT (grid coordinates)",
                        2: "UNIT (unit tag)",
                        3: "AUTO"
                    }.get(ability.get('target_type', 0), "UNKNOWN")
                    
                    abilities_info.append(
                        f"  [{idx}] {ability.get('name', 'Unknown')}"
                        f" - Type: {target_type_str}"
                    )
                
                if abilities_info:
                    available_abilities_str += (
                        f"\n{unit['unit_name']} (Tag: {unit['simplified_tag']}):\n" +
                        "\n".join(abilities_info) + "\n"
                    )

        # 修改示例部分，使用引号包裹字典字符串
        example_actions = """
        Example Ability Actions:
        - Quick cast: Ability: 1 -> 0 -> {'target_type': 0}
        - Point target: Ability: 2 -> 1 -> {'target_type': 1, 'position': [3, 4]}
        - Unit target: Ability: 13 -> 0 -> {'target_type': 2, 'target_unit': 14}
        """

        # 修改提示词，使用独立的示例部分
        prompt = f"""
        CRITICAL INSTRUCTION:
        Your primary skill is {primary_skill.get('name', 'None')}. All actions must implement this skill according to these steps:
        {skill_steps_str}

        Current Game State Analysis:
        Current game state:
        {units_info_str}

        {available_abilities_str}  # 添加技能信息

        {movement_system}

        DO NOT USE:
        - Pixel coordinates
        - Mathematical expressions
        - Invalid directions/coordinates
        
        Priority Targets Analysis:
        {important_units_response}
        
        Unit Capabilities:
        {unit_summary}
        
        Recent History:
        {format_history_for_prompt(self.history,history_length=self.history_length)}

        FINAL INSTRUCTION:
        Generate attack, movement, AND ability actions that STRICTLY implement the {skill_name} skill.
        Each action's reasoning must explain how it contributes to executing this skill.
        
        Use this format:
        ## Attack Actions ##
        Attack: [Unit Tag] -> [Target Tag]
        Reasoning: [Brief explanation]

        ## Move Actions ##
        Move: [Unit Tag] -> [x, y]
        Reasoning: [Brief explanation]

        ## Ability Actions ##
        Ability: [Unit Tag] -> [Ability Index] -> [Target Info]
        Reasoning: [Brief explanation]

        {example_actions}
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
        """为提示词格式化单位信息

        Args:
            unit_info: 包含单位信息的字典列表

        Returns:
            str: 格式化后的单位信息字符串
        """
        formatted_info = []
        for unit in unit_info:
            alliance = "Friendly" if unit['alliance'] == 1 else "Enemy"
            health_info = f"{unit['health']}/{unit['max_health']}"
            shield_info = f"{unit['shield']}/{unit['max_shield']}" if unit['max_shield'] > 0 else "No shields"
            energy_info = f"Energy: {unit['energy']}" if unit['energy'] > 0 else ""
            position_info = f"Position: [{unit['grid_position'][0]:.1f}, {unit['grid_position'][1]:.1f}]"
            unit_desc = (
                f"{unit['unit_name']} ({alliance}, Tag: {unit['simplified_tag']})\n"
                f"Health: {health_info}, Shields: {shield_info}\n"
                f"{energy_info}\n"
                f"{position_info}"
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
        """更新历史记录,包含技能规划信息"""
        # 确保移动动作格式一致
        move_actions = []
        if action['move']:
            for move in action['move']:
                if len(move) == 3:  # 如果是 (unit_tag, move_type, target) 格式
                    unit_tag, _, target = move
                    move_actions.append((unit_tag, target))
                elif len(move) == 2:  # 如果是 (unit_tag, target) 格式
                    move_actions.append(move)

        self.history.append({
            "step": self.step_count,
            "important_units": important_units,
            "attack_actions": action['attack'],
            "move_actions": move_actions,  # 使用处理后的移动动作
            "planned_skills": planned_skills,
            "move_type": self.move_type
        })

        if len(self.history) > self.history_length:
            self.history.pop(0)
            self.history.pop(0)

    def _validate_and_convert_coordinates(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """验证坐标是否在有效范围内"""
        if 0 <= x <= 9 and 0 <= y <= 9:
            return (x, y)
        return None

    def _should_update_assignments(self, observation) -> bool:
        """判断是否需要更新角色分配"""
        system_prompt = """You are a StarCraft II battle analyzer focusing on tactical situation assessment. 
        You need to determine if the current battlefield situation requires a role reassignment for units.
        Consider unit count changes, health status changes, and positional changes.
        Respond with 'true' if reassignment is needed, 'false' if not, followed by a brief explanation."""

        # 准备当前状态信息
        current_state = {
            'current_units': observation['unit_info'],
            'step_count': self.step_count,
            'current_assignments': self.current_assignments
        }

        user_prompt = f"""Current battlefield state:
        Step: {self.step_count}
        Units: {self.format_units_info_for_prompt(observation['unit_info'])}
        Current assignments: {self.current_assignments}

        Previous state (if available):
        Units: {self.format_units_info_for_prompt(getattr(self, 'previous_units', []))}

        Determine if role reassignment is needed based on:
        1. Unit count changes
        2. Significant health/shield changes
        3. Major positional changes
        4. Time since last update (consider updating every 30 steps)
        """

        update_decision_bot = TextChatbot(system_prompt=system_prompt, use_proxy=self.use_proxy)
        response = update_decision_bot.query(user_prompt)
        update_decision_bot.clear_history()

        # 解析响应
        should_update = response.lower().startswith('true')
        if should_update:
            self.previous_units = observation['unit_info']

        return should_update

    def _evaluate_battle_outcome(self, observation) -> Dict:
        """使用LLM评估当前战斗效果"""
        system_prompt = """You are a StarCraft II battle analyst specializing in combat effectiveness evaluation.
        Analyze the current battle situation and provide a detailed assessment in JSON format covering:
        1. Survival assessment (unit counts and health status)
        2. Combat efficiency (resource trades and unit exchanges)
        3. Objective completion (map control and mission goals)"""

        user_prompt = f"""Analyze the current battle situation:

        Friendly Units:
        {self.format_units_info_for_prompt([u for u in observation['unit_info'] if u['alliance'] == 1])}

        Enemy Units:
        {self.format_units_info_for_prompt([u for u in observation['unit_info'] if u['alliance'] != 1])}

        Current Assignments:
        {self.current_assignments}

        Previous State:
        {self._format_previous_state()}

        Provide a comprehensive battle assessment in the following JSON format:
        {{
            "survival_assessment": {{
                "friendly_units_count": <int>,
                "enemy_units_count": <int>,
                "friendly_health_percentage": <float>,
                "enemy_health_percentage": <float>
            }},
            "efficiency_assessment": {{
                "resource_efficiency": <float>,
                "exchange_ratio": <float>
            }},
            "objective_assessment": {{
                "map_control": <float>,
                "key_positions_control": <float>,
                "task_completion": <float>
            }}
        }}
        """

        assessment_bot = TextChatbot(system_prompt=system_prompt, use_proxy=self.use_proxy)
        response = assessment_bot.query(user_prompt)
        assessment_bot.clear_history()

        try:
            # 解析JSON响应
            assessment = json.loads(response)
            assessment['timestamp'] = self.step_count
            return assessment
        except json.JSONDecodeError:
            logger.error("Failed to parse battle assessment response")
            return self._generate_default_assessment()

    def _generate_default_assessment(self) -> Dict:
        """生成默认的评估结果"""
        return {
            'survival_assessment': {
                'friendly_units_count': 0,
                'enemy_units_count': 0,
                'friendly_health_percentage': 0.0,
                'enemy_health_percentage': 0.0
            },
            'efficiency_assessment': {
                'resource_efficiency': 1.0,
                'exchange_ratio': 1.0
            },
            'objective_assessment': {
                'map_control': 0.5,
                'key_positions_control': 0.0,
                'task_completion': 0.0
            },
            'timestamp': self.step_count
        }

    def _format_previous_state(self) -> str:
        """格式化上一个状态的信息"""
        if not hasattr(self, 'previous_units'):
            return "No previous state available"

        return f"""Previous unit counts:
        Friendly: {len([u for u in self.previous_units if u['alliance'] == 1])}
        Enemy: {len([u for u in self.previous_units if u['alliance'] != 1])}"""

    def _validate_ability_target(self, target_type, target_info, valid_friendly_tags, valid_enemy_tags):
        """验证技能目标的有效性"""
        try:
            if target_type == 0:  # QUICK cast
                return True
            
            elif target_type == 1:  # POINT target
                if 'position' not in target_info:
                    logger.warning("Missing position in POINT target ability")
                    return False
                x, y = target_info['position']
                valid = 0 <= x <= 9 and 0 <= y <= 9
                if not valid:
                    logger.warning(f"Invalid position coordinates: [{x}, {y}]")
                return valid
            
            elif target_type == 2:  # UNIT target
                if 'target_unit' not in target_info:
                    logger.warning("Missing target_unit in UNIT target ability")
                    return False
                target_unit = target_info['target_unit']
                ability_name = target_info.get('ability_name', '').upper()
                
                # 医疗和装载类技能只能对友军使用
                if any(keyword in ability_name for keyword in ['HEAL', 'LOAD', 'MEDIVAC']):
                    valid = target_unit in valid_friendly_tags
                    if not valid:
                        logger.warning(f"Invalid friendly target {target_unit} for {ability_name}")
                    return valid
                # 其他技能默认对敌人使用
                else:
                    valid = target_unit in valid_enemy_tags
                    if not valid:
                        logger.warning(f"Invalid enemy target {target_unit} for {ability_name}")
                    return valid
                
            logger.warning(f"Invalid target_type: {target_type}")
            return False
        
        except Exception as e:
            logger.error(f"Error validating ability target: {e}")
            return False
