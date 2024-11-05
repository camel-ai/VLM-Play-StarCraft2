import json
from pathlib import Path


def extract_unit_names(input_file_path):
    # 读取JSON文件
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 存储所有找到的unit_names
    unit_names = set()

    # 遍历所有steps
    for step in data['steps']:
        # 检查是否存在unit_info
        if 'unit_info' in step:
            # 从每个unit中提取unit_name
            for unit in step['unit_info']:
                if 'unit_name' in unit:
                    unit_names.add(unit['unit_name'])

    # 转换为列表并排序，确保输出一致性
    unit_names_list = sorted(list(unit_names))

    # 创建输出文件路径
    output_path = Path(input_file_path).parent / 'unit_names.json'

    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "unit_names": unit_names_list
        }, f, indent=2, ensure_ascii=False)

    return unit_names_list


# 使用示例
input_path = r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\run_env\log\TestAgent\20241024_194058\episode_6\logs_file\complete_data.json"
unit_names = extract_unit_names(input_path)
print("提取的单位名称:", unit_names)