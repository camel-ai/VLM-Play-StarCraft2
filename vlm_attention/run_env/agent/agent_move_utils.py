import re
import logging
from typing import List, Tuple, Dict, Any, Optional
import json
from vlm_attention.utils.call_vlm import MultimodalChatbot

from datetime import datetime
logger = logging.getLogger(__name__)
import os

"""

辅助函数和常量，用于处理VLM任务中的文本数据与格式化,标记任务
"""


def generate_unit_selection_prompt(available_units: List[str], important_units_response: str) -> str:
    return f"""
    Based on the image analysis and the available units in our database, select the most relevant units to query for more information. 
    Available units: {', '.join(available_units)}
    Image analysis result: {important_units_response}

    Please provide a list of unit names (comma-separated) that we should query from the database.
    """


def summarize_unit_info(unit_info: Dict[str, Dict]) -> str:
    summary_prompt = "Summarize the key information for the following StarCraft 2 units, including their strengths, weaknesses, and special abilities:\n\n"
    for unit, info in unit_info.items():
        summary_prompt += f"Unit: {unit}\n"
        summary_prompt += f"Description: {info.get('Description', 'N/A')}\n"
        summary_prompt += f"Strong against: {', '.join(info.get('Strong against', ['N/A']))}\n"
        summary_prompt += f"Weak against: {', '.join(info.get('Weak against', ['N/A']))}\n"
        summary_prompt += f"Special ability: {info.get('Ability', 'N/A')}\n\n"
    return summary_prompt


def generate_important_units_prompt() -> str:
    return """
    You are an AI assistant specialized in analyzing StarCraft II game states. Your task is to identify and explain 
    the most strategically important enemy units based on the provided game information and screenshot. 

    When analyzing, consider the following:
    1. Focus on units that significantly impact the battle due to their abilities or potential threat.
    2. Prioritize only those units that are key to the current game dynamics.
    3. Avoid selecting all enemy units; choose only the most crucial ones.

    For each important unit you identify, provide:
    - The unit's name
    - Its tag number
    - A detailed reason explaining its strategic importance in the current scenario

    Use the following format for each unit:

    ## Important Units ##
    Unit: [Unit Name]
    Tag: [Tag Number]
    Reason: [Explanation of why the unit is important]

    Repeat this format for each important unit. Do not include any other text or explanations outside of this format.
    """


def generate_decision_prompt() -> str:
    return """You are a StarCraft II expert focusing on unit micro-management. 
    Your task is to generate both attack and movement actions that follow the planned micro skills.

    CRITICAL RULES:
    1. Use ONLY valid unit tags provided in the unit information
    2. Friendly units can only attack enemy units
    3. Only friendly units can be given move commands
    4. Each unit can have only ONE action (either attack OR move)

    Your response MUST strictly follow this format:

    ## Attack Actions ##
    Attack: [Friendly Unit Tag] -> [Enemy Unit Tag]
    Reasoning: [Brief explanation]

    ## Move Actions ##
    Move: [Friendly Unit Tag] -> [x, y]
    Reasoning: [Brief explanation]

    Critical Format Rules:
    1. Use ONLY valid unit tags as shown in the unit information
    2. Grid coordinates must be integers from 0-9 (NOT pixel coordinates)
    3. Each unit should have exactly ONE action (either attack OR move)
    4. DO NOT use unit names in the actions, only their tag numbers

    Example Correct Format:
    ## Attack Actions ##
    Attack: 1 -> 8
    Reasoning: Target priority unit to reduce enemy damage output

    ## Move Actions ##
    Move: 2 -> [3, 4]
    Reasoning: Reposition to maintain safe attack range

    Remember:
    - Grid coordinates: [0,0] is top-left, [9,9] is bottom-right
    - Use ONLY grid coordinates (0-9), NOT pixel coordinates
    - Use ONLY numerical tags from the provided unit information
    """


def format_units_info(unit_info: List[Dict[str, Any]], predefined_tags: Dict[str, int],
                      unit_align_dict: Dict[str, str]) -> str:
    our_units = []
    enemy_units = []

    for unit in unit_info:
        try:
            tag_index = unit['tag_index']
            alliance = unit['alliance']

            unit_type = next((key.split(',')[0] for key, value in predefined_tags.items() if value == tag_index), None)

            if unit_type:
                unit_name = unit_align_dict.get(unit_type, "Unknown")
                unit_str = f"{unit_name}: Tag {tag_index}"

                if alliance == 1:  # Self
                    our_units.append(unit_str)
                elif alliance == 4:  # Enemy
                    enemy_units.append(unit_str)
                else:
                    logger.warning(f"Unknown alliance for unit: {unit}")
            else:
                logger.warning(f"Unknown unit type for tag: {tag_index}")

        except Exception as e:
            logger.error(f"Error processing unit: {unit}. Error: {e}", exc_info=True)

    return f"Our units:\n" + "\n".join(our_units) + "\n\nEnemy units:\n" + "\n".join(enemy_units)


def parse_vlm_response(response: str) -> List[int]:
    important_units = []
    unit_blocks = re.findall(r'## Important Units ##(.*?)(?=## Important Units ##|\Z)', response, re.DOTALL)

    for block in unit_blocks:
        lines = block.strip().split('\n')
        unit_name = ""
        tag = None
        for line in lines:
            if line.startswith("Unit:"):
                unit_name = line.split(":")[1].strip()
            elif line.startswith("Tag:"):
                tag_str = line.split(":")[1].strip()
                try:
                    # 尝试从字符串中提取数字
                    tag_match = re.search(r'\d+', tag_str)
                    if tag_match:
                        tag = int(tag_match.group())
                        important_units.append(tag)
                        logger.info(f"Successfully parsed tag {tag} for unit {unit_name}")
                    else:
                        logger.warning(f"No valid tag found in string: '{tag_str}' for unit {unit_name}")
                except ValueError as e:
                    logger.warning(f"Failed to parse tag for unit {unit_name}. Tag string: '{tag_str}'. Error: {e}")

        if tag is None:
            logger.warning(f"No valid tag found for unit: {unit_name}")

    return important_units


def parse_vlm_decision(response: str, move_type: str = 'grid') -> Dict[str, List[Tuple]]:
    """解析VLM的决策回复"""
    attack_actions = []
    move_actions = []

    # 解析攻击动作
    attack_pattern = re.compile(r'Attack:\s*(\d+)\s*->\s*(\d+)')
    for match in attack_pattern.finditer(response):
        try:
            attacker_tag = int(match.group(1))
            target_tag = int(match.group(2))
            attack_actions.append((attacker_tag, target_tag))
        except ValueError as e:
            logger.warning(f"Failed to parse attack tags: {e}")

    # 解析网格移动命令
    if move_type == 'grid':
        move_pattern = re.compile(r'Move:\s*(\d+)\s*->\s*\[(\d+)\s*,\s*(\d+)\]')
        for match in move_pattern.finditer(response):
            try:
                unit_tag = int(match.group(1))
                x = int(match.group(2))
                y = int(match.group(3))

                # 验证网格坐标是否在有效范围内
                if 0 <= x <= 9 and 0 <= y <= 9:
                    # 返回 (unit_tag, target) 格式
                    move_actions.append((unit_tag, [x, y]))
                else:
                    logger.warning(f"Invalid grid coordinates: [{x}, {y}]")
            except ValueError as e:
                logger.warning(f"Failed to parse grid move: {e}")

    else:  # SMAC movement
        move_pattern = re.compile(r'Move:\s*(\d+)\s*->\s*\[(\d+)\]')
        for match in move_pattern.finditer(response):
            try:
                unit_tag = int(match.group(1))
                direction = int(match.group(2))

                if 0 <= direction <= 3:
                    # 返回 (unit_tag, target) 格式
                    move_actions.append((unit_tag, [direction, 0]))
                else:
                    logger.warning(f"Invalid SMAC direction: {direction}")
            except ValueError as e:
                logger.warning(f"Failed to parse SMAC move: {e}")

    logger.info(f"Parsed attack actions: {attack_actions}")
    logger.info(f"Parsed move actions: {move_actions}")

    return {
        'attack': attack_actions,
        'move': move_actions
    }


def format_history_for_prompt(history: List[Dict[str, Any]], history_length: int) -> str:
    if not history:
        return "No previous steps available."

    formatted_history = []
    for entry in history:
        step_info = f"Step {entry['step']}:\n"

        # 保持原有的重要单位信息
        step_info += f"Important Units: {', '.join(map(str, entry['important_units']))}\n"

        # 攻击动作
        if entry['attack_actions']:
            step_info += "Attack Actions:\n"
            for attack in entry['attack_actions']:
                step_info += f"  Attack: {attack[0]} -> {attack[1]}\n"

        # 移动动作 - 修改这部分以适应新的格式
        if 'move_actions' in entry and entry['move_actions']:
            step_info += "Move Actions:\n"
            for move in entry['move_actions']:
                # 检查移动动作的格式
                if len(move) == 3:  # 如果是 (unit_tag, move_type, target) 格式
                    unit_tag, move_type, target = move
                    step_info += f"  Move: {unit_tag} -> {target}\n"
                elif len(move) == 2:  # 如果是 (unit_tag, target) 格式
                    unit_tag, target = move
                    step_info += f"  Move: {unit_tag} -> {target}\n"

        formatted_history.append(step_info)

    return "\n".join(formatted_history[-history_length:])


def generate_enhanced_unit_selection_prompt(available_units: List[str], important_units_response: str,
                                            text_observation: str) -> str:
    return f"""
    You are a StarCraft 2 strategist specializing in micro-management tasks. 
    We are currently in a micro-management scenario where all units are visible (no fog of war).

    Based on the image analysis and the available units in our database, select the most relevant units to query for more information. 
    Focus on units that are crucial for micro-management decisions.

    Available units: {', '.join(available_units)}

    Image analysis result: {important_units_response}

    Current game state observation:
    {text_observation}

    Please provide a list of unit names (comma-separated) that we should query from the database to assist in micro-management decision-making.
    """


def generate_unit_info_summary_prompt(unit_info: Dict[str, Dict]) -> str:
    summary_prompt = """
    You are a StarCraft 2 expert focusing on micro-management. 
    Summarize the key information for the following units, emphasizing aspects crucial for micro-management decision-making.
    Keep the summary concise and directly applicable to our current micro-management task.

    For each unit, briefly highlight:
    1. Its primary role in combat
    2. Key strengths in micro-management scenarios
    3. Critical weaknesses to be aware of
    4. Any special abilities that could be game-changing in micro-management

    Units to summarize:
    """
    for unit, info in unit_info.items():
        summary_prompt += f"\n{unit}:\n"
        summary_prompt += f"Description: {info.get('Description', 'N/A')}\n"
        summary_prompt += f"Strong against: {', '.join(info.get('Strong against', ['N/A']))}\n"
        summary_prompt += f"Weak against: {', '.join(info.get('Weak against', ['N/A']))}\n"
        summary_prompt += f"Special ability: {info.get('Ability', 'N/A')}\n"

    return summary_prompt


def generate_action_normalization_prompt(text_observation: str, unit_info: str,
                                         raw_action: Dict[str, List[Tuple]]) -> str:
    return f"""
    As a StarCraft 2 expert, review the current game state and generate optimal attack and movement actions for our units.

    Current game state:
    {text_observation}

    Unit information:
    {unit_info}

    Please provide both attack and movement actions for our units using the following format:
    ## Attack Actions ##
    Attack: [Attacker Tag] -> [Target Tag]
    Reasoning: [Brief explanation of the strategic choice]

    ## Move Actions ##
    Move: [Unit Tag] -> [x, y]
    Reasoning: [Brief explanation of the movement strategy]

    Ensure that:
    1. All attacker/moving unit tags correspond to our units.
    2. All attack target tags correspond to enemy units.
    3. Each unit should either attack OR move, not both.
    4. Movement coordinates must be within the 10x10 grid (0-9 for both x and y).
    5. All actions make strategic sense given the current game state.

    Example:
    ## Attack Actions ##
    Attack: 1 -> 9
    Reasoning: The Stalker targets the Ghost to neutralize its high damage potential and disruptive abilities.

    ## Move Actions ##
    Move: 2 -> [3, 4]
    Reasoning: The Phoenix repositions to a safer grid position while maintaining attack range.

    Remember: Grid coordinates [0,0] is top-left, [9,9] is bottom-right.
    """
def normalization_system_prompt() -> str:
    return """
    You are a StarCraft 2 expert tasked with normalizing unit actions. 
    Your output MUST strictly follow this exact format:

    ## Attack Actions ##
    Attack: [Tag Number] -> [Tag Number]
    Reasoning: [Brief explanation]

    ## Move Actions ##
    Move: [Tag Number] -> [x, y]
    Reasoning: [Brief explanation]

    Critical Format Rules:
    1. Use ONLY tag numbers (like 1, 2, 3), NOT unit names
    2. Attack actions must be in format: "Attack: X -> Y"
    3. Move actions must be in format: "Move: X -> [x, y]"
    4. Grid coordinates must be integers from 0-9
    5. Each unit must have exactly ONE action
    6. Include both section headers exactly as shown
    7. DO NOT use unit names like "Zealot_1" or "Marine_2"

    Example Correct Output:
    ## Attack Actions ##
    Attack: 1 -> 7
    Reasoning: Eliminate high-value target

    ## Move Actions ##
    Move: 2 -> [5, 4]
    Reasoning: Maintain safe firing position

    Remember: All coordinates must be within 0-9 range.
    """

class VLMPlanner:
    def __init__(self, save_dir: str, replan_each_step: bool = False):
        """初始化VLM规划器

        Args:
            save_dir: 保存目录
            replan_each_step: 是否每步重新规划
        """
        self.save_dir = save_dir
        self.replan_each_step = replan_each_step
        self.current_plan = None
        self.plan_history = []
        os.makedirs(save_dir, exist_ok=True)

    def _get_planner_system_prompt(self) -> str:
        return """You are a StarCraft II micro-management expert. Your task is to analyze the current combat situation 
        and plan appropriate micro-management tactics based on the screenshot and unit information.

        Focus on micro-management skills:
        1. Focus Fire: Concentrating damage on specific targets
        2. Kiting: Hit and run tactics
        3. Shield/Health Management: Managing unit durability
        4. Formation Control: Unit positioning

        Your response MUST strictly follow this JSON-like format:

        ### MICRO PLAN ###
        {
            "primary_skill": {
                "name": "Focus Fire",
                "description": "Concentrating damage on specific targets",
                "implementation_steps": [
                    "Step 1: Select highest priority target",
                    "Step 2: Command all units to attack the same target"
                ]
            },
            "secondary_skills": [
                {
                    "name": "Kiting",
                    "description": "Hit and run tactics",
                    "when_to_use": "When enemy units are approaching"
                },
                {
                    "name": "Formation Control",
                    "description": "Positioning units strategically",
                    "when_to_use": "When need to maximize damage output while minimizing damage taken"
                }
            ]
        }
        ### END PLAN ###

        Ensure your response maintains this exact format with the headers and JSON structure.
        """

    def plan(self, observation: Dict[str, Any], image_path: str,use_proxy:bool=False) -> Dict[str, Any]:
        """生成微操技能规划"""
        # 如果不是每步重新规划且已有计划,直接返回当前计划
        if not self.replan_each_step and self.current_plan is not None:
            return self.current_plan

        # 构建规划提示词
        planning_prompt = self._generate_planning_prompt(observation)

        # 使用camel框架
        planner_bot = MultimodalChatbot(
            system_prompt=self._get_planner_system_prompt(),
            use_proxy=use_proxy
        )
        response = planner_bot.query(planning_prompt, image_path=image_path)
        planner_bot.clear_history()

        # 解析响应获取技能规划
        skills = self._parse_skills(response)

        # 更新历史
        self._update_plan_history(skills, observation)

        # 保存规划记录
        self._save_planning_io(planning_prompt, response, skills, image_path)

        return skills

    def _update_plan_history(self, skills: Dict[str, Any], observation: Dict[str, Any]):
        """更新规划历史"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'skills': skills,
            'situation': observation.get('text', ''),
            'units_count': len([u for u in observation['unit_info'] if u['alliance'] == 1]),
            'enemy_count': len([u for u in observation['unit_info'] if u['alliance'] != 1])
        }
        self.plan_history.append(history_entry)
        self.current_plan = skills

    def _format_plan_history(self) -> str:
        """格式化规划历史"""
        if not self.plan_history:
            return "No previous planning history."

        history_lines = []
        for entry in self.plan_history[-3:]:  # 只显示最近3条历史
            history_lines.append(
                f"Previous plan ({entry['timestamp']}):\n"
                f"- Primary Skill: {entry['skills'].get('primary', {}).get('name')}\n"
                f"- Situation: {entry['situation']}\n"
                f"- Units: {entry['units_count']} friendly vs {entry['enemy_count']} enemy\n"
            )
        return "\n".join(history_lines)
    def _generate_planning_prompt(self, observation: Dict[str, Any]) -> str:
        """生成规划提示词"""
        history_info = self._format_plan_history()

        return f"""Analyze the current StarCraft II combat situation and plan appropriate micro-management skills.

Current situation:
{observation.get('text', 'No text observation available.')}

Previous planning history:
{history_info}

Based on the screenshot, situation and planning history:
1. What should be our primary micro skill?
2. What supporting skills might be useful?
3. How should we implement these skills?

Focus only on micro-management skills and tactics, without specifying exact unit targets or movement commands.
"""

    def _parse_skills(self, response: str) -> Dict[str, Any]:
        """解析VLM响应中的技能规划"""
        try:
            # 提取JSON部分
            json_match = re.search(r'### MICRO PLAN ###\s*({.*?})\s*### END PLAN ###', response, re.DOTALL)

            if not json_match:
                logger.warning("Failed to find JSON content in response")
                return {}

            json_str = json_match.group(1)

            # 解析JSON
            skills_data = json.loads(json_str)

            # 转换为期望的格式
            return {
                'primary': {
                    'name': skills_data['primary_skill']['name'],
                    'description': skills_data['primary_skill']['description'],
                    'steps': skills_data['primary_skill']['implementation_steps']
                },
                'secondary': [
                    {
                        'name': skill['name'],
                        'description': skill['description'],
                        'condition': skill['when_to_use']
                    }
                    for skill in skills_data['secondary_skills']
                ]
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing skills: {e}")
            return {}

    def _save_planning_io(self, prompt: str, response: str, skills: Dict[str, Any], image_path: str):
        """保存规划的输入输出"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"planning_io_{timestamp}.json")

        data = {
            "prompt": prompt,
            "response": response,
            "parsed_skills": skills,
            "image_path": image_path,
            "timestamp": timestamp,
            "parsing_success": bool(skills)  # 添加解析是否成功的标志
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def format_units_info_for_prompt(self, unit_info: List[Dict[str, Any]]) -> str:
        formatted_info = []
        for unit in unit_info:
            alliance = "Friendly" if unit['alliance'] == 1 else "Enemy"
            health_info = f"{unit['health']}/{unit['max_health']}"
            shield_info = f"{unit['shield']}/{unit['max_shield']}" if unit['max_shield'] > 0 else "No shields"
            energy_info = f"Energy: {unit['energy']}" if unit['energy'] > 0 else ""

            # 直接使用grid_position
            grid_pos = unit['grid_position']

            unit_desc = (
                f"{unit['unit_name']} ({alliance}, Tag: {unit['simplified_tag']})\n"
                f"Health: {health_info}, Shields: {shield_info}\n"
                f"{energy_info}\n"
                f"Grid Position: [{grid_pos[0]}, {grid_pos[1]}]"
            )
            formatted_info.append(unit_desc)

        return "\n\n".join(formatted_info)



