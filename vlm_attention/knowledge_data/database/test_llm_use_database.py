import logging
from typing import List, Dict
from sc2_unit_database import SC2UnitDatabase
from vlm_attention.utils.call_vlm import MultimodalChatbot, TextChatbot

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_unit_info_from_database(db: SC2UnitDatabase, unit_names: List[str]) -> Dict[str, Dict]:
    """从数据库获取指定单位的信息"""
    unit_info = {}
    for unit in unit_names:
        info = db.get_unit_info(unit)
        if info:
            unit_info[unit] = info
        else:
            logger.warning(f"未找到单位信息: {unit}")
    return unit_info


def summarize_unit_info(text_bot: TextChatbot, unit_info: Dict[str, Dict]) -> str:
    """使用语言模型总结单位信息"""
    summary_prompt = "请总结以下StarCraft 2单位的关键信息，包括它们的优势、劣势和特殊能力：\n\n"
    for unit, info in unit_info.items():
        summary_prompt += f"单位：{unit}\n"
        summary_prompt += f"描述：{info.get('Description', 'N/A')}\n"
        summary_prompt += f"优势：{', '.join(info.get('Strong against', ['N/A']))}\n"
        summary_prompt += f"劣势：{', '.join(info.get('Weak against', ['N/A']))}\n"
        summary_prompt += f"特殊能力：{info.get('Ability', 'N/A')}\n\n"

    summary = text_bot.query("You are a StarCraft 2 expert. Summarize the following unit information concisely.",
                             summary_prompt)
    return summary


def main():
    # 初始化数据库和聊天机器人
    db = SC2UnitDatabase("sc2_unit_data_index.yaml")
    multi_bot = MultimodalChatbot(model_name="gpt-4o")
    text_bot = TextChatbot(model_name="gpt-4o")

    # 步骤1：使用多模态模型分析图像
    image_path = r"D:\pythoncode\vlm_attention_starcraft2-main\vlm_attention\utils\9a64ffa2e5ae3562902565b7cf9bb34.png"
    starcraft_system_prompt = """
    This is a StarCraft2 micro mini game.
    
        Our units are:
    - Stalker (Tag: 1)
    - Zealot (Tag: 2)
    - Sentry (Tag: 3)
    - Immortal (Tag: 4)
    - Archon (Tag: 5)
    
    Enemy units are:
    - Siege Tank Sieged (Tag: 6)
    - Reaper (Tag: 7)
    - Ghost (Tag: 8)
    - Marine (Tag: 9)
    - Marauder (Tag: 10)
    - Medivac (Tag: 11)
    - Hellbat (Tag: 12)
    - Banshee (Tag: 13)
    - Viking Assault (Tag: 14)
    
    Analyze the current StarCraft II game state shown in the screenshot and identify the most strategically important 
    enemy units. For each selected unit, provide the unit's name, its tag, and a detailed reason why it is considered 
    crucial in this scenario. Focus on units that significantly impact the battle due to their abilities or potential 
    threat. Avoid selecting all enemy units; prioritize only those that are key to the current game dynamics. Here is the 
    format for your response:
    
    Unit: [Unit Name].
    Tag: [Tag Number].
    Reason: [Explanation of why the unit is important].
    """
    response = multi_bot.query(starcraft_system_prompt, "Identify the most important enemy units.", image_path)
    logger.info(f"多模态模型分析结果：\n{response}")

    # 步骤2和3：从数据库获取单位信息并使用语言模型进行查询选择
    available_units = db.get_all_unit_names()
    unit_selection_prompt = f"""
    Based on the image analysis and the available units in our database, select the most relevant units to query for more information. 
    Available units: {', '.join(available_units)}
    Image analysis result: {response}

    Please provide a list of unit names (comma-separated) that we should query from the database.
    """
    units_to_query = text_bot.query("You are a StarCraft 2 strategist.", unit_selection_prompt)
    logger.info(f"选择查询的单位：{units_to_query}")

    # 步骤4：查询数据库并使用语言模型总结信息
    unit_list = [unit.strip() for unit in units_to_query.split(',')]
    unit_info = get_unit_info_from_database(db, unit_list)
    summary = summarize_unit_info(text_bot, unit_info)
    logger.info(f"单位信息总结：\n{summary}")

    # 步骤5：根据总结提供战略建议
    strategy_prompt = f"""
    Based on the following information:
    1. Image analysis: {response}
    2. Unit information summary: {summary}

    Provide a comprehensive strategy for our units to counter the enemy composition and exploit their weaknesses.
    """
    strategy = text_bot.query("You are an expert StarCraft 2 strategist.", strategy_prompt)
    logger.info(f"战略建议：\n{strategy}")


if __name__ == "__main__":
    main()