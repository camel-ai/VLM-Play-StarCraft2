import re
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

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
    return """
    You are an AI assistant specialized in StarCraft II micro-management strategy. Your task is to suggest optimal attack and movement actions for our units based on the current game state.

    Your response MUST strictly follow this format:

    ## Attack Actions ##
    Attack: [Tag Number] -> [Tag Number]
    Reasoning: [Brief explanation]

    ## Move Actions ##
    Move: [Tag Number] -> [x, y]
    Reasoning: [Brief explanation]

    Critical Format Rules:
    1. Use ONLY tag numbers (like 1, 2, 3), not unit names (no Zealot_1, Marine_2 etc)
    2. Each action must be on its own line
    3. Include both section headers exactly as shown
    4. Grid coordinates must be integers from 0-9
    5. Each unit should have exactly ONE action (either attack OR move)
    6. DO NOT use unit names in the actions, only their tag numbers

    Example Correct Format:
    ## Attack Actions ##
    Attack: 1 -> 8
    Reasoning: Target priority unit to reduce enemy damage output

    ## Move Actions ##
    Move: 2 -> [3, 4]
    Reasoning: Reposition to maintain safe attack range

    Remember:
    - Grid coordinates: [0,0] is top-left, [9,9] is bottom-right
    - Use ONLY numerical tags, not unit names
    - Include reasoning for each action
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


def parse_vlm_decision(response: str) -> Dict[str, List[Tuple]]:
    """解析VLM的决策回复,更严格的格式处理

    Args:
        response: VLM的原始回复文本
        期望格式:
        ## Attack Actions ##
        Attack: 1 -> 8
        Reasoning: ...

        ## Move Actions ##
        Move: 2 -> [3, 4]
        Reasoning: ...

    Returns:
        Dict包含两个键:
        - 'attack': List[Tuple[int, int]] 攻击动作列表 (攻击者tag, 目标tag)
        - 'move': List[Tuple[int, int, List[int]]] 移动动作列表 (单位tag, move_type, [x, y]坐标)
    """
    attack_actions = []
    move_actions = []

    # 严格检查必需的章节头
    if not all(header in response for header in ["## Attack Actions ##", "## Move Actions ##"]):
        logger.warning("Response missing required section headers")
        return {'attack': attack_actions, 'move': move_actions}

    # 解析攻击动作
    attack_pattern = re.compile(r'Attack:\s*(\d+)\s*->\s*(\d+)')
    for match in attack_pattern.finditer(response):
        try:
            attacker_tag = int(match.group(1))
            target_tag = int(match.group(2))
            attack_actions.append((attacker_tag, target_tag))
            logger.debug(f"Parsed attack action: {attacker_tag} -> {target_tag}")
        except ValueError as e:
            logger.warning(f"Failed to parse attack tags: {e}")

    # 解析移动动作
    move_pattern = re.compile(r'Move:\s*(\d+)\s*->\s*\[(\d+)\s*,\s*(\d+)\]')
    for match in move_pattern.finditer(response):
        try:
            unit_tag = int(match.group(1))
            x = int(match.group(2))
            y = int(match.group(3))
            if 0 <= x <= 9 and 0 <= y <= 9:  # 验证坐标范围
                move_actions.append((unit_tag, 1, [x, y])) # 1表示基于网格的移动
                logger.debug(f"Parsed move action: {unit_tag} -> [1, [{x}, {y}]]")
            else:
                logger.warning(f"Invalid grid coordinates: [{x}, {y}]")
        except ValueError as e:
            logger.warning(f"Failed to parse move action: {e}")

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

        # 移动动作
        if 'move_actions' in entry and entry['move_actions']:
            step_info += "Move Actions:\n"
            for move_type, unit_tag, target in entry['move_actions']:
                if move_type == 1:  # grid-based movement
                    step_info += f"  Move: {unit_tag} -> {target}\n"

        formatted_history.append(step_info)

    return "\n".join(formatted_history[-history_length:])  # 保持显示最近3步的历史


def generate_enhanced_unit_selection_prompt(available_units: List[str], important_units_response: str,
                                            text_observation: str) -> str:
    return f"""
    You are a StarCraft 2 strategist specializing in micro-management tasks for the Protoss race. 
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
    You are a StarCraft 2 expert focusing on Protoss micro-management. 
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