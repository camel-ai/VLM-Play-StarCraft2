import json
import logging
import os
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np


from vlm_attention.env.config import COLORS, get_unit_name
from vlm_attention.knowledge_data.database.sc2_unit_database import SC2UnitDatabase
from vlm_attention.run_env.agent.agent_move_utils import (
    summarize_unit_info, generate_important_units_prompt,
    generate_decision_prompt, format_units_info, parse_vlm_response, parse_vlm_decision,
    format_history_for_prompt, generate_enhanced_unit_selection_prompt,
    generate_unit_info_summary_prompt, generate_action_normalization_prompt, normalization_system_prompt
)
from vlm_attention.run_env.utils import _annotate_units_on_image, draw_grid_with_labels
from vlm_attention.utils.call_vlm import MultimodalChatbot, TextChatbot
from vlm_attention.run_env.agent.role_assignment import RoleAssignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VLMAgent:
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, grid_size: Tuple[int, int] = (10, 10),
                 use_self_attention: bool = False, use_rag: bool = False, history_length: int = 3):
        """
        初始化VLMAgent代理。
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
        self.history: List[Dict[str, Any]] = []
        self.draw_grid = draw_grid
        self.annotate_units = annotate_units
        self.grid_size = grid_size
        self.step_count = 0
        self.use_self_attention = use_self_attention
        self.use_rag = use_rag
        self.history_length = history_length

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

        # 初始化角色分配模块
        self.role_assignment = RoleAssignment(config_path)
        self.current_assignments = {}

        logger.info(f"VLMAgent initialized. Data will be saved in: {self.save_dir}")
        logger.info(f"Using YAML file: {yaml_path}")
        logger.info(f"Self-attention: {'ON' if self.use_self_attention else 'OFF'}")
        logger.info(f"RAG: {'ON' if self.use_rag else 'OFF'}")

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, List[Tuple]]:
        """获取动作决策,包含攻击和移动"""
        self.step_count += 1
        self.text_observation = observation["text"]
        logger.info(f"Processing step {self.step_count}")
        logger.info(f"Text observation: {self.text_observation}")

        # 保存该step的unit info
        self._save_step_data(observation)

        # 保存原始图像
        if self.save_original_images:
            self._save_image(observation['image'], self.original_images_dir, "original")

        # 处理并保存第一次图像注释
        first_annotation_path = self._process_and_save_image(observation, self.first_annotation_dir, "first_annotation",
                                                             annotate_all=True)

        # 如果启用了自注意力机制，识别重要单位
        if self.use_self_attention:
            important_units_response = self._identify_important_units(observation, first_annotation_path)
            self.important_units = parse_vlm_response(important_units_response)
            logger.info(f"Identified important units: {self.important_units}")
            decision_image_path = self._process_and_save_image(observation, self.second_annotation_dir,
                                                               "second_annotation", annotate_all=False)
        else:
            self.important_units = []
            important_units_response = "Self-attention is disabled."
            decision_image_path = first_annotation_path

        # RAG处理
        unit_summary = ""
        if self.use_rag:
            units_to_query = ([unit for unit in observation['unit_info']
                               if unit['alliance'] == 1 or unit['simplified_tag'] in self.important_units]
                              if self.use_self_attention else observation['unit_info'])

            unit_info = self._get_unit_info_from_database(units_to_query)
            unit_summary_system_prompt = "You are a StarCraft 2 expert focusing on Protoss micro-management."
            unit_summary_user_prompt = generate_unit_info_summary_prompt(unit_info)
            unit_summary_bot = TextChatbot(system_prompt=unit_summary_system_prompt, use_proxy=True)
            unit_summary = unit_summary_bot.query(unit_summary_user_prompt)
            unit_summary_bot.clear_history()
            self._save_vlm_io(unit_summary_user_prompt, unit_summary, "unit_summary")
            logger.info(f"单位信息总结：\n{unit_summary}")

        # 生成决策
        decision_system_prompt = generate_decision_prompt()
        decision_user_prompt = self._generate_decision_prompt(observation, unit_summary, important_units_response)
        raw_decision_bot = MultimodalChatbot(system_prompt=decision_system_prompt, use_proxy=True)
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
        normalized_action = {'attack': [], 'move': []}
        for attempt in range(max_retries):
            normalized_bot = TextChatbot(system_prompt=normalization_system_prompt(), use_proxy=True)
            normalized_action_response = normalized_bot.query(normalization_user_prompt)
            normalized_bot.clear_history()
            self._save_vlm_io(normalization_user_prompt, normalized_action_response,
                              f"normalized_decision_attempt_{attempt + 1}")

            normalized_action = parse_vlm_decision(normalized_action_response)
            # 只要有任何有效动作就接受
            if normalized_action['attack'] or normalized_action['move']:
                break
            elif attempt < max_retries - 1:
                logger.warning(f"Failed to parse actions on attempt {attempt + 1}. Retrying...")
            else:
                logger.error("Failed to parse actions after maximum retries. Returning empty actions.")

        # 更新历史记录
        self._update_history(self.important_units, normalized_action)

        # 在合适的时机更新角色分配
        if self._should_update_assignments(observation):
            # 生成初始分配方案
            new_assignments = self.role_assignment.initial_assignment(observation)
            
            # 执行对比学习
            tasks = list(new_assignments.values())
            task_similarities = self.role_assignment.contrastive_learning(tasks)
            
            # 根据战斗结果进行反思和调整
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

    def _identify_important_units(self, observation: Dict[str, Any], image_path: str) -> str:
        """识别重要单位"""
        system_prompt = generate_important_units_prompt()
        units_info_str = self.format_units_info_for_prompt(observation['unit_info'])

        user_input = f"""
            Analyze the current StarCraft II game state based on the following information:

            Screenshot observation:
            {observation.get("text", "No text observation available.")}

            Units information:
            {units_info_str}

            Previous steps information:
            {format_history_for_prompt(self.history, self.history_length)}

            Based on this information, identify and explain the most strategically important enemy units.
            """

        """
        使用camel 框架
        """
        important_unit_bot = MultimodalChatbot(system_prompt=system_prompt, use_proxy=True)
        important_units_response = important_unit_bot.query(user_input, image_path=image_path)
        important_unit_bot.clear_history()
        """
        使用camel 框架结束
        """
        self._save_vlm_io(user_input, important_units_response, "important_units")

        return important_units_response

    def _generate_decision_prompt(self, observation: Dict[str, Any], unit_summary: str,
                                  important_units_response: str) -> str:
        """生成决策提示,包含攻击和移动决策"""
        units_info_str = self.format_units_info_for_prompt(observation['unit_info'])

        prompt = f"""
            Analyze the current StarCraft II game state and suggest both attack and movement actions for our units:

            Screenshot observation:
            {self.text_observation}

            Units information:
            {units_info_str}

            Movement System:
            - The map is divided into a 10x10 grid (0-9 for both x and y coordinates)
            - Origin (0,0) is at the top-left corner
            - X increases to the right, Y increases downward
            - Each unit can either attack an enemy unit or move to a grid position

            Required Output Format:
            ## Attack Actions ##
            [Attacker Unit Name (Tag)] attacks [Target Unit Name (Tag)]

            ## Move Actions ##
            [Unit Name (Tag)] moves to grid [x, y]

            Example:
            ## Attack Actions ##
            Stalker_1 (Tag: 1) attacks Zealot_2 (Tag: 8)

            ## Move Actions ##
            Stalker_2 (Tag: 2) moves to grid [3, 4]
            """

        if self.use_self_attention:
            prompt += f"""
                Important enemy units analysis:
                {important_units_response}
                """

        if self.use_rag:
            prompt += f"""
                Unit information summary:
                {unit_summary}
                """

        prompt += f"""
            Previous steps information:
            {format_history_for_prompt(self.history, self.history_length)}

            Based on this information and micro-management principles, provide BOTH attack and movement actions for our units.
            Each unit should either attack or move, not both.
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

    def _update_history(self, important_units: List[int], actions: Dict[str, List[Tuple]]):
        """更新包含攻击和移动动作的历史记录"""
        self.history.append({
            "step": self.step_count,
            "important_units": important_units,
            "attack_actions": actions['attack'],
            "move_actions": actions['move']
        })
        if len(self.history) > self.history_length:
            self.history.pop(0)

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
        Units: {self._format_units_for_prompt(observation['unit_info'])}
        Current assignments: {self.current_assignments}
        
        Previous state (if available):
        Units: {self._format_units_for_prompt(getattr(self, 'previous_units', []))}
        
        Determine if role reassignment is needed based on:
        1. Unit count changes
        2. Significant health/shield changes
        3. Major positional changes
        4. Time since last update (consider updating every 30 steps)
        """
        
        update_decision_bot = TextChatbot(system_prompt=system_prompt, use_proxy=True)
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
        {self._format_units_for_prompt([u for u in observation['unit_info'] if u['alliance'] == 1])}
        
        Enemy Units:
        {self._format_units_for_prompt([u for u in observation['unit_info'] if u['alliance'] != 1])}
        
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
        
        assessment_bot = TextChatbot(system_prompt=system_prompt, use_proxy=True)
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

    def _format_units_for_prompt(self, units: List[Dict]) -> str:
        """格式化单位信息用于prompt"""
        formatted_units = []
        for unit in units:
            unit_info = (
                f"{unit['unit_name']} (Tag: {unit['simplified_tag']}):\n"
                f"- Health: {unit['health']}/{unit['max_health']}\n"
                f"- Shield: {unit['shield']}/{unit['max_shield']}\n"
                f"- Position: [{unit['position'][0]:.1f}, {unit['position'][1]:.1f}]"
            )
            formatted_units.append(unit_info)
        return "\n".join(formatted_units)

    def _format_previous_state(self) -> str:
        """格式化上一个状态的信息"""
        if not hasattr(self, 'previous_units'):
            return "No previous state available"
        
        return f"""Previous unit counts:
        Friendly: {len([u for u in self.previous_units if u['alliance'] == 1])}
        Enemy: {len([u for u in self.previous_units if u['alliance'] != 1])}"""

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
