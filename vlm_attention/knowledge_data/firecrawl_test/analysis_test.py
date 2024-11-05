from bs4 import BeautifulSoup
import json

def extract_info(soup, field_name):
    field = soup.find('div', class_='infobox-cell-2', string=field_name)
    if field:
        return field.find_next_sibling('div').text.strip()
    return "Not found"

def extract_list(soup, field_name):
    field = soup.find('div', class_='infobox-cell-2', string=field_name)
    if field:
        items = field.find_next_sibling('div').find_all('li')
        return [item.text.strip() for item in items]
    return []

def extract_full_description(soup):
    description_section = soup.find('span', class_='mw-headline', id='Description')
    if description_section:
        description = ""
        for sibling in description_section.parent.next_siblings:
            if sibling.name == 'p':
                description += sibling.text.strip() + " "
            elif sibling.name == 'h2':
                break
        return description.strip()
    return "Not found"

# 读取JSON文件
with open('D:/pythoncode/vlm_attention_starcraft2-main/vlm_attention/knowledge_data/firecrawl_test/sc2_unit_info/Protoss_colossus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取HTML内容
html_content = data['html']

# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(html_content, 'html.parser')

# 提取所有信息
unit_info = {
    "Type": extract_info(soup, "Type:"),
    "Description": extract_full_description(soup),
    "Built From": extract_info(soup, "Built From:"),
    "Requirements": extract_info(soup, "Requirements:"),
    "Cost": extract_info(soup, "Cost:"),
    "Hotkey": extract_info(soup, "Hotkey:"),
    "Attack": {
        "Targets": extract_info(soup, "Targets:"),
        "Damage": extract_info(soup, "Damage:"),
        "DPS": extract_info(soup, "DPS:"),
        "Cooldown": extract_info(soup, "Cooldown:"),
        "Bonus": extract_info(soup, "Bonus:"),
        "Bonus DPS": extract_info(soup, "Bonus DPS:"),
        "Range": extract_info(soup, "Range:")
    },
    "Unit stats": {
        "Defense": extract_info(soup, "Defense:"),
        "Attributes": extract_info(soup, "Attributes:"),
        "Sight": extract_info(soup, "Sight:"),
        "Speed": extract_info(soup, "Speed:"),
        "Cargo size": extract_info(soup, "Cargo size:")
    },
    "Strong against": extract_list(soup, "Strong against:"),
    "Weak against": extract_list(soup, "Weak against:")
}
# 提取ability
ability_section = soup.find('span', class_='mw-headline', id='Ability')
if ability_section:
    ability_div = ability_section.find_next('div', class_='wikitable')
    ability_name = ability_div.find('div', id='Cliff_Walk').text.strip()
    ability_description = ability_div.find('div', style='clear:both').text.strip()
    unit_info["Ability"] = f"{ability_name}: {ability_description}"
else:
    unit_info["Ability"] = "No ability found"

# 提取upgrade
upgrade_section = soup.find('span', class_='mw-headline', id='Upgrade')
if upgrade_section:
    upgrade_div = upgrade_section.find_next('div', class_='wikitable')
    upgrade_name = upgrade_div.find('div', id='Extended_Thermal_Lance').text.strip()
    upgrade_details = upgrade_div.find_all('div', style='padding:2px 10px')
    upgrade_description = upgrade_div.find('div', style='clear:both').text.strip()
    unit_info["Upgrade"] = {
        "Name": upgrade_name,
        "Details": [detail.text.strip() for detail in upgrade_details],
        "Description": upgrade_description
    }
else:
    unit_info["Upgrade"] = "No upgrade found"

# 提取Competitive Usage部分
competitive_usage = {}
competitive_usage_section = soup.find('span', class_='mw-headline', id='Competitive_Usage')
if competitive_usage_section:
    current_section = None
    for elem in competitive_usage_section.parent.next_siblings:
        if elem.name == 'h3':
            current_section = elem.text.strip()
            competitive_usage[current_section] = ""
        elif elem.name == 'p' and current_section:
            competitive_usage[current_section] += elem.text.strip() + " "
        elif elem.name == 'h2':
            break
unit_info["Competitive Usage"] = competitive_usage

# 输出结果
print(json.dumps(unit_info, indent=2))