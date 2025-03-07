from typing import Dict, List, Tuple
import logging
import json
from vlm_attention.utils.call_vlm import TextChatbot
import time
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class RoleAssignment:
    def __init__(self, config_path: str, use_proxy: bool = False):
        """初始化RoleAssignment"""
        logger.info(f"Initializing RoleAssignment with config path: {config_path}")
        self.config_path = config_path
        self.previous_assignments = {}
        self.task_embeddings = {}
        self.use_proxy = use_proxy
        self.max_retries = 3
        self.retry_delay = 2
        
        # 添加日志保存相关
        self.save_dir = os.path.dirname(config_path)
        self.role_io_dir = os.path.join(self.save_dir, "role_io")
        os.makedirs(self.role_io_dir, exist_ok=True)

    def _generate_situation_prompt(self, observation: Dict) -> Tuple[str, str]:
        """生成描述当前战场情势的prompt"""
        system_prompt = """You are a StarCraft II tactical advisor specializing in unit role assignment.
        Analyze the battlefield situation and assign appropriate roles to friendly units.
        Consider unit types, positions, capabilities, and the overall strategic context.
        Provide assignments in JSON format with unit tags as keys and role details as values."""

        # Format friendly and enemy units separately
        friendly_units = [unit for unit in observation.get('unit_info', []) if unit['alliance'] == 1]
        enemy_units = [unit for unit in observation.get('unit_info', []) if unit['alliance'] != 1]

        # Create formatted unit strings
        friendly_units_str = "\n".join([
            f"- {unit['unit_name']} (Tag: {unit['simplified_tag']}) at position [{unit['grid_position'][0]}, {unit['grid_position'][1]}]"
            for unit in friendly_units
        ])

        enemy_units_str = "\n".join([
            f"- {unit['unit_name']} (Tag: {unit['simplified_tag']}) at position [{unit['grid_position'][0]}, {unit['grid_position'][1]}]"
            for unit in enemy_units
        ])

        user_prompt = f"""Current battlefield situation:

Friendly Units:
{friendly_units_str}

Enemy Units:
{enemy_units_str}

Terrain Information:
{observation.get('text', 'Standard map terrain')}

Please assign roles to our units considering:
1. Unit combat capabilities (attack range, damage, speed)
2. Strategic positions and map control
3. Enemy unit composition and positions
4. Required tactical roles (Attacker, Defender, Scout, Support)

Provide the assignment in this JSON format:
{{
    "<unit_tag>": {{
        "role": "<role_name>",
        "primary_task": "<main_task>",
        "secondary_task": "<backup_task>",
        "target_position": [x, y],
        "priority_targets": ["<enemy_unit_tag>", ...],
        "coordination_group": "<group_id>"
    }},
    ...
}}
"""
        return system_prompt, user_prompt

    def _generate_contrastive_prompt(self, tasks: List[str]) -> Tuple[str, str]:
        """生成用于对比学习的prompt"""
        system_prompt = """You are a StarCraft II tactical analyst specializing in comparing and evaluating combat tasks.
        Analyze the similarity between different tactical tasks and provide detailed comparisons.
        Consider tactical purposes, execution requirements, and coordination needs."""

        user_prompt = f"""Compare the following tactical tasks:
        Tasks: {tasks}

        For each pair of tasks, provide a similarity analysis in JSON format:
        {{
            "task_pairs": [
                {{
                    "task1": "<task_name>",
                    "task2": "<task_name>",
                    "similarity_score": <0-1>,
                    "comparison_metrics": {{
                        "tactical_purpose": <0-1>,
                        "execution_difficulty": <0-1>,
                        "risk_level": <0-1>,
                        "coordination_requirement": <0-1>
                    }},
                    "explanation": "<detailed_analysis>"
                }},
                ...
            ]
        }}
        """
        return system_prompt, user_prompt

    def _generate_reflection_prompt(self,
                                    current_assignments: Dict,
                                    battle_outcome: Dict) -> Tuple[str, str]:
        """生成用于反思的prompt"""
        system_prompt = """You are a StarCraft II strategic advisor specializing in battle analysis and tactical adjustment.
        Review the current role assignments and battle outcomes to suggest optimizations.
        Provide detailed analysis and specific adjustment recommendations."""

        user_prompt = f"""Review the following battle information:

        Current Role Assignments:
        {json.dumps(current_assignments, indent=2)}

        Battle Outcomes:
        {json.dumps(battle_outcome, indent=2)}

        Provide a comprehensive analysis and adjustment plan in JSON format:
        {{
            "assessment": {{
                "effectiveness_rating": <0-1>,
                "key_observations": ["<observation>", ...],
                "identified_issues": ["<issue>", ...]
            }},
            "adjustments": {{
                "<unit_tag>": {{
                    "current_role": "<role>",
                    "suggested_role": "<role>",
                    "adjustment_reason": "<explanation>",
                    "priority": <1-5>
                }},
                ...
            }},
            "tactical_recommendations": ["<recommendation>", ...]
        }}
        """
        return system_prompt, user_prompt

    def _save_role_io(self, prefix: str, input_data: Dict, output_data: Dict):
        """保存角色分配的输入输出数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.role_io_dir, f"{prefix}_{timestamp}.json")
        
        data = {
            "timestamp": timestamp,
            "input": input_data,
            "output": output_data
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {prefix} I/O data to {filename}")
        except Exception as e:
            logger.error(f"Error saving {prefix} I/O data: {str(e)}")

    def initial_assignment(self, observation: Dict) -> Dict:
        """生成初始任务分配方案"""
        logger.info("Starting initial role assignment generation")
        system_prompt, user_prompt = self._generate_situation_prompt(observation)

        for attempt in range(self.max_retries):
            try:
                assignment_bot = TextChatbot(
                    system_prompt=system_prompt, 
                    use_proxy=self.use_proxy
                )
                response = assignment_bot.query(user_prompt)
                assignment_bot.clear_history()

                assignments = json.loads(response)
                self._validate_assignments(assignments)
                
                # 保存输入输出数据
                self._save_role_io(
                    "initial_assignment",
                    {"observation": observation, "prompt": user_prompt},
                    {"assignments": assignments, "response": response}
                )
                
                logger.info(f"Successfully generated initial assignments: {assignments}")
                return assignments
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                logger.error("All attempts failed for initial assignment")
                return {}

    def contrastive_learning(self, tasks: List[str]) -> Dict:
        """执行对比学习分析"""
        logger.info(f"Starting contrastive learning for tasks: {tasks}")
        system_prompt, user_prompt = self._generate_contrastive_prompt(tasks)

        for attempt in range(self.max_retries):
            try:
                contrastive_bot = TextChatbot(
                    system_prompt=system_prompt, 
                    use_proxy=self.use_proxy
                )
                response = contrastive_bot.query(user_prompt)
                contrastive_bot.clear_history()

                analysis = json.loads(response)
                self._update_task_embeddings(analysis)
                logger.info(f"Completed contrastive learning analysis: {analysis}")
                return analysis
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                logger.error("All attempts failed for contrastive learning")
                return {"task_pairs": []}

    def reflect_and_adjust(self, current_assignments: Dict, battle_outcome: Dict) -> Dict:
        """执行反思并调整任务分配"""
        logger.info("Starting reflection and adjustment process")
        logger.info(f"Current assignments: {current_assignments}")
        logger.info(f"Battle outcome: {battle_outcome}")

        system_prompt, user_prompt = self._generate_reflection_prompt(
            current_assignments, battle_outcome)

        for attempt in range(self.max_retries):
            try:
                reflection_bot = TextChatbot(
                    system_prompt=system_prompt, 
                    use_proxy=self.use_proxy
                )
                response = reflection_bot.query(user_prompt)
                reflection_bot.clear_history()

                adjustments = json.loads(response)
                new_assignments = self._apply_adjustments(current_assignments, adjustments)
                
                # 保存输入输出数据
                self._save_role_io(
                    "reflection_adjustment",
                    {
                        "current_assignments": current_assignments,
                        "battle_outcome": battle_outcome,
                        "prompt": user_prompt
                    },
                    {
                        "adjustments": adjustments,
                        "new_assignments": new_assignments,
                        "response": response
                    }
                )
                
                logger.info(f"Successfully adjusted assignments: {new_assignments}")
                return new_assignments
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                logger.error("All attempts failed for reflection and adjustment")
                return current_assignments

    def _format_units(self, units: List[Dict]) -> str:
        """格式化单位信息用于prompt"""
        logger.info("Formatting unit information for prompt")
        formatted_units = []
        for unit in units:
            unit_info = (
                f"{unit.get('unit_name', 'Unknown')} (Tag: {unit.get('simplified_tag', 'N/A')}):\n"
                f"- Health: {unit.get('health', 0)}/{unit.get('max_health', 0)}\n"
                f"- Shield: {unit.get('shield', 0)}/{unit.get('max_shield', 0)}\n"
                f"- Position: [{unit.get('grid_position', [0, 0])[0]:.1f}, "
                f"{unit.get('grid_position', [0, 0])[1]:.1f}]"
            )
            formatted_units.append(unit_info)
            logger.info(f"Formatted unit info: {unit_info}")
        return "\n".join(formatted_units)

    def _validate_assignments(self, assignments: Dict) -> None:
        """验证分配方案的格式和内容"""
        required_fields = ['role', 'primary_task', 'target_position']
        for unit_tag, assignment in assignments.items():
            missing_fields = [field for field in required_fields if field not in assignment]
            if missing_fields:
                logger.warning(f"Assignment for unit {unit_tag} missing fields: {missing_fields}")
            else:
                logger.info(f"Validated assignment for unit {unit_tag}: {assignment}")

    def _update_task_embeddings(self, analysis: Dict) -> None:
        """更新任务相似度映射"""
        logger.info("Updating task embeddings")
        for pair in analysis.get('task_pairs', []):
            task1 = pair.get('task1')
            task2 = pair.get('task2')
            similarity = pair.get('similarity_score', 0)

            if task1 and task2:
                if task1 not in self.task_embeddings:
                    self.task_embeddings[task1] = {}
                if task2 not in self.task_embeddings:
                    self.task_embeddings[task2] = {}

                self.task_embeddings[task1][task2] = similarity
                self.task_embeddings[task2][task1] = similarity
                logger.info(f"Updated similarity between {task1} and {task2}: {similarity}")

    def _apply_adjustments(self, current_assignments: Dict, adjustments: Dict) -> Dict:
        """应用反思后的调整建议"""
        logger.info("Starting to apply role adjustments")
        new_assignments = current_assignments.copy()

        for unit_tag, adjustment in adjustments.get('adjustments', {}).items():
            if unit_tag in new_assignments and adjustment.get('suggested_role'):
                old_role = new_assignments[unit_tag]['role']
                new_role = adjustment['suggested_role']
                new_assignments[unit_tag]['role'] = new_role
                logger.info(f"Unit {unit_tag} role changed: {old_role} -> {new_role}")

        return new_assignments
