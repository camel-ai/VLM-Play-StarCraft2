from typing import Dict, List, Tuple
import logging
import json
from vlm_attention.utils.call_vlm import call_llm, TextChatbot

logger = logging.getLogger(__name__)

class RoleAssignment:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.previous_assignments = {}
        self.task_embeddings = {}
        
    def _generate_situation_prompt(self, observation: Dict) -> Tuple[str, str]:
        """生成描述当前战场情势的prompt"""
        system_prompt = """You are a StarCraft II tactical advisor specializing in unit role assignment.
        Analyze the battlefield situation and assign appropriate roles to friendly units.
        Consider unit types, positions, capabilities, and the overall strategic context.
        Provide assignments in JSON format with unit tags as keys and role details as values."""
        
        user_prompt = f"""Current battlefield situation:

        Friendly Units:
        {self._format_units(observation.get('friendly_units', []))}
        
        Enemy Units:
        {self._format_units(observation.get('enemy_units', []))}
        
        Terrain Information:
        {observation.get('terrain_info', 'Standard map terrain')}
        
        Generate a role assignment plan considering:
        1. Unit combat capabilities (attack range, damage, speed)
        2. Strategic positions and map control
        3. Enemy unit composition and positions
        4. Required tactical roles (Attacker, Defender, Scout, Support)
        
        Provide the assignment in this JSON format:
        {
            "<unit_tag>": {
                "role": "<role_name>",
                "primary_task": "<main_task>",
                "secondary_task": "<backup_task>",
                "target_position": [x, y],
                "priority_targets": ["<enemy_unit_tag>", ...],
                "coordination_group": "<group_id>"
            },
            ...
        }
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

    def initial_assignment(self, observation: Dict) -> Dict:
        """生成初始任务分配方案"""
        system_prompt, user_prompt = self._generate_situation_prompt(observation)
        
        assignment_bot = TextChatbot(system_prompt=system_prompt, use_proxy=True)
        response = assignment_bot.query(user_prompt)
        assignment_bot.clear_history()
        
        try:
            assignments = json.loads(response)
            self._validate_assignments(assignments)
            return assignments
        except json.JSONDecodeError:
            logger.error("Failed to parse initial assignment response")
            return {}

    def contrastive_learning(self, tasks: List[str]) -> Dict:
        """执行对比学习分析"""
        system_prompt, user_prompt = self._generate_contrastive_prompt(tasks)
        
        contrastive_bot = TextChatbot(system_prompt=system_prompt, use_proxy=True)
        response = contrastive_bot.query(user_prompt)
        contrastive_bot.clear_history()
        
        try:
            analysis = json.loads(response)
            self._update_task_embeddings(analysis)
            return analysis
        except json.JSONDecodeError:
            logger.error("Failed to parse contrastive learning response")
            return {"task_pairs": []}

    def reflect_and_adjust(self, 
                          current_assignments: Dict,
                          battle_outcome: Dict) -> Dict:
        """执行反思并调整任务分配"""
        system_prompt, user_prompt = self._generate_reflection_prompt(
            current_assignments, battle_outcome)
        
        reflection_bot = TextChatbot(system_prompt=system_prompt, use_proxy=True)
        response = reflection_bot.query(user_prompt)
        reflection_bot.clear_history()
        
        try:
            adjustments = json.loads(response)
            return self._apply_adjustments(current_assignments, adjustments)
        except json.JSONDecodeError:
            logger.error("Failed to parse reflection response")
            return current_assignments

    def _format_units(self, units: List[Dict]) -> str:
        """格式化单位信息用于prompt"""
        formatted_units = []
        for unit in units:
            unit_info = (
                f"{unit.get('unit_name', 'Unknown')} (Tag: {unit.get('simplified_tag', 'N/A')}):\n"
                f"- Health: {unit.get('health', 0)}/{unit.get('max_health', 0)}\n"
                f"- Shield: {unit.get('shield', 0)}/{unit.get('max_shield', 0)}\n"
                f"- Position: [{unit.get('position', [0, 0])[0]:.1f}, "
                f"{unit.get('position', [0, 0])[1]:.1f}]"
            )
            formatted_units.append(unit_info)
        return "\n".join(formatted_units)

    def _validate_assignments(self, assignments: Dict) -> None:
        """验证分配方案的格式和内容"""
        required_fields = ['role', 'primary_task', 'target_position']
        for unit_tag, assignment in assignments.items():
            missing_fields = [field for field in required_fields if field not in assignment]
            if missing_fields:
                logger.warning(f"Assignment for unit {unit_tag} missing fields: {missing_fields}")

    def _update_task_embeddings(self, analysis: Dict) -> None:
        """更新任务相似度映射"""
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

    def _apply_adjustments(self, current_assignments: Dict, adjustments: Dict) -> Dict:
        """应用反思后的调整建议"""
        new_assignments = current_assignments.copy()
        
        for unit_tag, adjustment in adjustments.get('adjustments', {}).items():
            if unit_tag in new_assignments and adjustment.get('suggested_role'):
                new_assignments[unit_tag]['role'] = adjustment['suggested_role']
                
        return new_assignments 