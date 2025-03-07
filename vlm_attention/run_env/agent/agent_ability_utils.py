import re
import logging
from typing import List, Tuple, Dict, Any, Optional
import json
from vlm_attention.utils.call_vlm import MultimodalChatbot,TextChatbot

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
    return """You are an AI assistant specialized in analyzing StarCraft II game states. Your task is to identify and explain 
    the most strategically important enemy units based on the provided game information and screenshot. 

    CRITICAL RULES:
    1. You MUST identify at least 2-3 important enemy units
    2. For each important unit, you MUST provide:
       - Unit name
       - Tag number
       - Detailed reason for importance
    3. Focus on units that:
       - Have tactical advantages (high health/shields, special abilities)
       - Are in threatening positions
       - Could disrupt our planned strategy
    4. Your response MUST strictly follow this format:

    ## Important Units ##
    Unit: [Unit Name]
    Tag: [Tag Number]
    Reason: [Detailed tactical explanation]

    ## Important Units ##
    Unit: [Unit Name]
    Tag: [Tag Number]
    Reason: [Detailed tactical explanation]

    [Repeat for each important unit]

    EXAMPLE RESPONSE:
    ## Important Units ##
    Unit: Zealot
    Tag: 7
    Reason: Highest shield value (50/50) and positioned closest to our units at [0,1], immediate threat to our Reapers

    ## Important Units ##
    Unit: Zealot
    Tag: 12
    Reason: Full shields and blocking key escape route at [1,1], could trap our units if not dealt with

    Remember:
    - ALWAYS include the exact tag numbers
    - ALWAYS explain tactical importance
    - ALWAYS identify at least 2-3 units
    - ALWAYS use the exact format with headers
    """


def generate_decision_prompt(move_type: str = 'grid') -> str:
    """生成决策提示词"""
    base_prompt = """You are a StarCraft II battle commander. Generate optimal actions for each unit following these rules:

1. Each unit can have ONE action only (attack, move, OR ability)
2. Only use valid unit tags from the current state
3. Abilities must use correct indices and target types
4. All actions must contribute to the current strategy
"""

    if move_type == 'grid':
        base_prompt += """
Movement uses grid coordinates:
- Valid range: 0-9 for both x and y
- [0,0] is top-left corner
- [9,9] is bottom-right corner
Example: Move: 1 -> [5, 3]
"""
    else:
        base_prompt += """
Movement uses directional numbers:
- Valid range: 0-7 (clockwise)
- 0=North, 2=East, 4=South, 6=West
Example: Move: 1 -> 2
"""

    base_prompt += """
Format your response as:

## Attack Actions ##
Attack: [Unit Tag] -> [Target Tag]
Reasoning: [Brief explanation]

## Move Actions ##
Move: [Unit Tag] -> [coordinates/direction]
Reasoning: [Brief explanation]

## Ability Actions ##
Ability: [Unit Tag] -> [Ability Index] -> {target_info}
Reasoning: [Brief explanation]

Example Ability Actions:
- Quick: Ability: 1 -> 0 -> {target_type: 0}
- Point: Ability: 2 -> 1 -> {target_type: 1, position: [3, 4]}
- Unit: Ability: 13 -> 2 -> {target_type: 2, target_unit: 14}
"""

    return base_prompt


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
    """Parse VLM response to extract important unit tags"""
    important_units = []
    unit_blocks = re.findall(r'## Important Units ##(.*?)(?=## Important Units ##|\Z)', response, re.DOTALL)
    
    if not unit_blocks:
        logger.warning("No important units found in response")
        return important_units
        
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
                    tag_match = re.search(r'\d+', tag_str)
                    if tag_match:
                        tag = int(tag_match.group())
                        important_units.append(tag)
                        logger.info(f"Found important unit: {unit_name} with tag {tag}")
                    else:
                        logger.warning(f"No valid tag found in string: '{tag_str}' for unit {unit_name}")
                except ValueError as e:
                    logger.warning(f"Failed to parse tag for unit {unit_name}. Tag string: '{tag_str}'. Error: {e}")

    if not important_units:
        logger.warning("No valid unit tags found in response")
    else:
        logger.info(f"Total important units found: {len(important_units)}")
        
    return important_units


def parse_vlm_decision(response: str, move_type: str = 'grid') -> Dict[str, List[Tuple]]:
    """解析VLM的决策回复"""
    attack_actions = []
    move_actions = []
    ability_actions = []

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
                    move_actions.append((unit_tag, [direction, 0]))
                else:
                    logger.warning(f"Invalid SMAC direction: {direction}")
            except ValueError as e:
                logger.warning(f"Failed to parse SMAC move: {e}")

    # 解析技能动作 - 支持多种格式
    ability_patterns = [
        # 格式1: Ability: 1 -> 0 -> {}
        re.compile(r'Ability:\s*(\d+)\s*->\s*(\d+)\s*->\s*\{\}'),
        # 格式2: Ability: 13 -> 0 -> QUICK
        re.compile(r'Ability:\s*(\d+)\s*->\s*(\d+)\s*->\s*QUICK'),
        # 格式3: Ability: 2 -> 1 -> POINT[x,y]
        re.compile(r'Ability:\s*(\d+)\s*->\s*(\d+)\s*->\s*POINT\[(\d+)\s*,\s*(\d+)\]'),
        # 格式4: Ability: 13 -> 2 -> UNIT14
        re.compile(r'Ability:\s*(\d+)\s*->\s*(\d+)\s*->\s*UNIT(\d+)'),
        # 格式5: Ability: 1 -> 0 -> {target_type: 0}
        re.compile(r'Ability:\s*(\d+)\s*->\s*(\d+)\s*->\s*\{[^}]+\}')
    ]

    for pattern in ability_patterns:
        for match in pattern.finditer(response):
            try:
                unit_tag = int(match.group(1))
                ability_idx = int(match.group(2))
                target_info = {'target_type': 0}  # 默认为QUICK类型

                if len(match.groups()) > 2:
                    if 'POINT' in pattern.pattern:
                        target_info = {
                            'target_type': 1,
                            'position': [int(match.group(3)), int(match.group(4))]
                        }
                    elif 'UNIT' in pattern.pattern:
                        target_info = {
                            'target_type': 2,
                            'target_unit': int(match.group(3))
                        }
                    elif match.group(3).startswith('{'):
                        try:
                            target_info = eval(match.group(3))
                        except:
                            logger.warning(f"Failed to parse target info dict: {match.group(3)}")

                ability_actions.append((unit_tag, ability_idx, target_info))
                logger.info(f"Parsed ability action: {unit_tag} -> {ability_idx} -> {target_info}")

            except Exception as e:
                logger.warning(f"Failed to parse ability action: {e}")
                continue

    logger.info(f"Parsed attack actions: {attack_actions}")
    logger.info(f"Parsed move actions: {move_actions}")
    logger.info(f"Parsed ability actions: {ability_actions}")

    return {
        'attack': attack_actions,
        'move': move_actions,
        'ability': ability_actions
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

        # 移动动作
        if 'move_actions' in entry and entry['move_actions']:
            step_info += "Move Actions:\n"
            for move in entry['move_actions']:
                if len(move) == 3:  # 如果是 (unit_tag, move_type, target) 格式
                    unit_tag, move_type, target = move
                    step_info += f"  Move: {unit_tag} -> {target}\n"
                elif len(move) == 2:  # 如果是 (unit_tag, target) 格式
                    unit_tag, target = move
                    step_info += f"  Move: {unit_tag} -> {target}\n"

        # 技能动作
        if 'ability_actions' in entry and entry['ability_actions']:
            step_info += "Ability Actions:\n"
            for ability in entry['ability_actions']:
                unit_tag, ability_index, target_info = ability
                target_type = target_info['target_type']
                if target_type == 0:
                    target_str = "QUICK"
                elif target_type == 1:
                    target_str = f"POINT{target_info['position']}"
                elif target_type == 2:
                    target_str = f"UNIT{target_info['target_unit']}"
                else:
                    target_str = "AUTO"
                step_info += f"  Ability: {unit_tag} -> {ability_index} -> {target_str}\n"

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
    As a StarCraft 2 expert, review the current game state and generate optimal attack, movement, and ability actions for our units.

    Current game state:
    {text_observation}

    Unit information:
    {unit_info}

    Please provide attack, movement, and ability actions for our units using the following format:
    ## Attack Actions ##
    Attack: [Attacker Tag] -> [Target Tag]
    Reasoning: [Brief explanation of the strategic choice]

    ## Move Actions ##
    Move: [Unit Tag] -> [x, y]
    Reasoning: [Brief explanation of the movement strategy]

    ## Ability Actions ##
    Ability: [Unit Tag] -> [Ability Index] -> [Target Info]
    Reasoning: [Brief explanation of the ability usage]

    Ensure that:
    1. All attacker/moving/ability-using unit tags correspond to our units
    2. All attack target tags correspond to enemy units
    3. Each unit should have exactly ONE action (attack, move, or ability)
    4. Movement coordinates must be within the 10x10 grid (0-9 for both x and y)
    5. Ability indices must be valid for the unit
    6. Ability target info must match the ability's target type:
       - QUICK (0): No target needed
       - POINT (1): Grid coordinates [x, y]
       - UNIT (2): Target unit tag
       - AUTO (3): No target needed
    7. All actions make strategic sense given the current game state

    Example:
    ## Attack Actions ##
    Attack: 1 -> 9
    Reasoning: The Stalker targets the Ghost to neutralize its high damage potential

    ## Move Actions ##
    Move: 2 -> [3, 4]
    Reasoning: The Phoenix repositions to a safer grid position

    ## Ability Actions ##
    Ability: 3 -> 0 -> POINT[5, 6]
    Reasoning: Cast Psi Storm on clustered enemy units

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

    ## Ability Actions ##
    Ability: [Tag Number] -> [Ability Index] -> [Target Info]
    Reasoning: [Brief explanation]

    Critical Format Rules:
    1. Use ONLY tag numbers (like 1, 2, 3), NOT unit names
    2. Attack actions must be in format: "Attack: X -> Y"
    3. Move actions must be in format: "Move: X -> [x, y]"
    4. Ability actions must be in format: "Ability: X -> Y -> TARGET"
    5. Grid coordinates must be integers from 0-9
    6. Each unit must have exactly ONE action
    7. Include all section headers exactly as shown
    8. DO NOT use unit names like "Zealot_1" or "Marine_2"
    9. Target Info must be one of:
       - QUICK for no-target abilities
       - POINT[x,y] for location-targeted abilities
       - UNITz for unit-targeted abilities (z is target tag)
       - AUTO for automatic abilities

    Example Correct Output:
    ## Attack Actions ##
    Attack: 1 -> 7
    Reasoning: Eliminate high-value target

    ## Move Actions ##
    Move: 2 -> [5, 4]
    Reasoning: Maintain safe firing position

    ## Ability Actions ##
    Ability: 3 -> 1 -> POINT[6, 7]
    Reasoning: Use area effect ability on enemy cluster

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

def generate_important_units_normalization_prompt() -> str:
    """生成用于规范化important units响应的系统提示词"""
    return """You are a StarCraft II combat analyzer specializing in standardizing unit importance assessments.
    Your task is to take raw analysis and convert it into a properly formatted list of important units.

    CRITICAL FORMAT RULES:
    1. You MUST identify exactly 2-3 important enemy units
    2. You MUST use the exact format:
       ## Important Units ##
       Unit: [Unit Name]
       Tag: [Tag Number]
       Reason: [Detailed tactical explanation]
    3. Each reason MUST include:
       - Unit's current status (health/shields)
       - Position-based threat assessment
       - Tactical importance
    4. You MUST only use valid unit tags from the provided information

    EXAMPLE CORRECT FORMAT:
    ## Important Units ##
    Unit: Zealot
    Tag: 7
    Reason: Highest shield value (50/50) and positioned at [0,1], immediate threat to our Reapers

    ## Important Units ##
    Unit: Zealot
    Tag: 12
    Reason: Full shields and blocking key escape route at [1,1], could trap our units

    DO NOT include any other text or explanations outside this format."""

def normalize_important_units(raw_response: str, observation: Dict[str, Any], use_proxy: bool = False) -> str:
    """规范化important units的响应格式"""
    system_prompt = generate_important_units_normalization_prompt()
    
    # 准备单位信息
    enemy_units = [unit for unit in observation['unit_info'] if unit['alliance'] != 1]
    enemy_info = "\n".join([
        f"Enemy {unit['unit_name']} (Tag: {unit['simplified_tag']}):\n"
        f"- Health: {unit['health']}/{unit['max_health']}\n"
        f"- Shields: {unit.get('shield', 0)}/{unit.get('max_shield', 0)}\n"
        f"- Position: [{unit['grid_position'][0]}, {unit['grid_position'][1]}]"
        for unit in enemy_units
    ])

    user_prompt = f"""Normalize the following unit importance analysis into the required format.

Available Enemy Units:
{enemy_info}

Raw Analysis:
{raw_response}

Requirements:
1. Select 2-3 most important units from the available enemies
2. Use exact format from system prompt
3. Include detailed tactical reasons
4. Use only valid unit tags
5. Focus on units with tactical advantages"""

    normalization_bot = TextChatbot(system_prompt=system_prompt, use_proxy=use_proxy)
    normalized_response = normalization_bot.query(user_prompt)
    normalization_bot.clear_history()

    return normalized_response

def format_observation_for_prompt(observation: Dict, move_type: str = 'grid') -> str:
    """格式化观测信息为简洁的提示词格式"""
    
    # 基础信息部分
    prompt = """Current Battle State:

FRIENDLY UNITS:
"""
    # 格式化友方单位信息
    for unit in observation['unit_info']:
        if unit['alliance'] == 1:  # 友方单位
            abilities_str = ""
            if 'abilities' in unit:
                abilities_str = "\n    Abilities: " + ", ".join(
                    f"[{idx}]{ability['name']}({ability['target_type']})" 
                    for idx, ability in enumerate(unit['abilities'])
                )
            
            unit_str = f"""- {unit['unit_type']}_{unit['tag_index']} (Tag: {unit['simplified_tag']})
    Health: {unit['health']}/{unit['max_health']}
    Position: {unit['position']}""" + abilities_str
            prompt += unit_str + "\n"

    prompt += "\nENEMY UNITS:\n"
    
    # 格式化敌方单位信息
    for unit in observation['unit_info']:
        if unit['alliance'] == 4:  # 敌方单位
            unit_str = f"""- {unit['unit_type']}_{unit['tag_index']} (Tag: {unit['simplified_tag']})
    Health: {unit['health']}/{unit['max_health']}
    Position: {unit['position']}"""
            prompt += unit_str + "\n"

    # 根据move_type添加不同的移动说明
    if move_type == 'grid':
        prompt += """
MOVEMENT RULES:
- Use grid coordinates [0-9]
- [0,0] is top-left, [9,9] is bottom-right
- Format: Move: [Unit Tag] -> [x, y]
"""
    else:  # smac
        prompt += """
MOVEMENT RULES:
- Use relative direction [0-7]
- 0=North, clockwise to 7=Northwest
- Format: Move: [Unit Tag] -> [direction]
"""

    prompt += """
ACTION FORMATS:
## Attack Actions ##
Attack: [Unit Tag] -> [Enemy Tag]

## Move Actions ##
Move: [Unit Tag] -> [coordinates/direction]

## Ability Actions ##
Ability: [Unit Tag] -> [Ability Index] -> {target_info}
- QUICK (0): {target_type: 0}
- POINT (1): {target_type: 1, position: [x, y]}
- UNIT (2): {target_type: 2, target_unit: tag}
"""

    return prompt