import os
import yaml
"""
每次安装时请注意初始化yaml配置文件
"""

def scan_unit_files(folder_path):
    unit_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            parts = filename.split('_')
            if len(parts) >= 2:
                race = parts[0]
                unit_name = '_'.join(parts[1:]).rsplit('.', 1)[0]
                file_path = os.path.join(folder_path, filename)
                if race not in unit_data:
                    unit_data[race] = {}
                unit_data[race][unit_name] = {
                    'file_name': filename,
                    'file_path': file_path
                }
    return unit_data


def save_to_yaml(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


def main():
    folder_path = r"C:\python_code\vlm_attention_starcraft2\vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info_processed"# 处理后的文件夹路径,请设置绝对地址.
    """
    Example:
    vlm_attention/knowledge_data/firecrawl_test/sc2_unit_info_processed
    """
    output_file = "sc2_unit_data_index.yaml"

    unit_data = scan_unit_files(folder_path)
    save_to_yaml(unit_data, output_file)

    total_units = sum(len(race_units) for race_units in unit_data.values())
    print(f"Scanned {total_units} unit files.")
    print(f"Data index saved to {output_file}")

    # Print a brief summary
    for race, units in unit_data.items():
        print(f"\n{race} units ({len(units)}):")
        for unit_name in units:
            print(f"  - {unit_name}")


if __name__ == "__main__":
    main()