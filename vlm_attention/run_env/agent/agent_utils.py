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
    return """You are a StarCraft II expert focusing on unit micro-management. 
    Analyze the game state and provide attack actions for our units.

    Your response MUST use the following format:

    ## Actions ##
    Attack: [Unit Tag Number] -> [Target Tag Number]
    Reasoning: [Brief explanation]

    Example:
    ## Actions ##
    Attack: 1 -> 8
    Reasoning: Targeting the Medivac to eliminate enemy healing capabilities.

    Important:
    - Use ONLY unit tag numbers (1, 2, 3, etc), not unit names
    - Each action must be on a new line
    - Include the '## Actions ##' header exactly as shown
    - Provide reasoning for each action
    - Focus on the most strategically important targets first
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
    """Parse VLM response with more flexible format matching"""
    attack_actions = []

    # Define multiple possible section headers
    action_headers = ["## Actions ##", "## Attack Actions ##"]
    response_sections = response.split("\n")

    # Find the action section
    action_section = []
    in_action_section = False
    for line in response_sections:
        if any(header in line for header in action_headers):
            in_action_section = True
            continue
        if in_action_section and line.strip() and not line.startswith("##"):
            action_section.append(line)
        elif in_action_section and line.startswith("##"):
            break

    if not action_section:
        logger.warning("VLM response does not contain a valid actions section")
        return {'attack': attack_actions, 'move': []}

    # Parse actions with more flexible pattern matching
    attack_pattern = re.compile(r'Attack:[\s\n]*([A-Za-z0-9_]+)[\s\n]*->[\s\n]*([A-Za-z0-9_]+)', re.IGNORECASE)
    matches = attack_pattern.findall('\n'.join(action_section))

    for attacker, target in matches:
        try:
            # Extract tag numbers from unit names or use direct numbers
            attacker_tag = int(attacker.split('_')[1]) if '_' in attacker else int(attacker)
            target_tag = int(target.split('_')[1]) if '_' in target else int(target)
            attack_actions.append((attacker_tag, target_tag))
            logger.debug(f"Parsed attack action: {attacker}({attacker_tag}) -> {target}({target_tag})")
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse attack action: {attacker} -> {target}: {e}")

    return {
        'attack': attack_actions,
        'move': []
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
def normalization_system_prompt():
    return """
    You are a StarCraft 2 expert tasked with reviewing and normalizing actions. 
                Your output must strictly follow this format:
                ## Actions ##
                Attack: [Attacker Tag] -> [Target Tag]
                Reasoning: [Brief explanation]

                Repeat this format for each attack action. Ensure all attacker tags are our units and all target tags are enemy units.

                Example output:
                ## Actions ##
                Attack: 1 -> 9
                Reasoning: The Stalker focuses on the Ghost due to its high damage potential and disabling abilities.
            """