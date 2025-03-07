import os
import shutil
from typing import List, Dict


"""
check the unit name in the json file from the firecrawl_test,
and standardize the unit name to the pysc2 standard name
"""

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")


def standardize_unit_files(source_path: str, target_path: str, standard_names: List[str], dry_run: bool = True) -> None:
    """
    根据PySC2的标准名称规范化单位文件名并保存到新目录

    Args:
        source_path: 源JSON文件所在文件夹路径
        target_path: 目标文件夹路径
        standard_names: PySC2标准单位名称列表
        dry_run: 是否只预览更改
    """
    # 单位到种族的映射
    race_mapping = {
        'Archon': 'Protoss',
        'Banshee': 'Terran',
        'Ghost': 'Terran',
        'Hellbat': 'Terran',
        'Immortal': 'Protoss',
        'Marauder': 'Terran',
        'Marine': 'Terran',
        'Medivac': 'Terran',
        'Phoenix': 'Protoss',
        'Reaper': 'Terran',
        'Stalker': 'Protoss',
        'Viking Assault': 'Terran',
        'Viking Fighter': 'Terran',
        'Zealot': 'Protoss'
    }

    # 确保目标目录存在
    if not dry_run:
        ensure_dir(target_path)

    # 获取现有文件列表
    existing_files = os.listdir(source_path)

    print(f"{'预览更改' if dry_run else '执行更改'}:")
    print("-" * 50)

    def find_file(unit_name: str, race: str) -> str:
        """辅助函数：查找匹配的文件"""
        unit_simple = unit_name.lower().replace(' ', '_')
        race_simple = race.lower()

        for file in existing_files:
            file_simple = file.lower()
            if file_simple.startswith(f"{race_simple}_{unit_simple}") or \
                    file_simple.startswith(f"{race_simple}_{unit_simple.replace('_', '')}"):
                return file
            # 特殊处理viking的情况
            if unit_simple in ['viking_assault', 'viking_fighter'] and 'viking.json' in file_simple:
                return file
        return None

    # 处理每个标准单位名称
    for unit_name in standard_names:
        race = race_mapping.get(unit_name)
        if not race:
            print(f"警告: 未找到 {unit_name} 的种族信息")
            continue

        # 构建新的文件名 (保持原始大小写)
        new_filename = f"{race}_{unit_name}.json"

        # 特殊处理Viking的变体
        if unit_name in ['Viking Assault', 'Viking Fighter']:
            viking_file = next((f for f in existing_files if 'viking' in f.lower()), None)
            if viking_file:
                source_file = os.path.join(source_path, viking_file)
                target_file = os.path.join(target_path, new_filename)

                if dry_run:
                    print(f"将从 {viking_file} 复制创建 {new_filename}")
                else:
                    try:
                        shutil.copy2(source_file, target_file)
                        print(f"成功复制: {viking_file} -> {new_filename}")
                    except Exception as e:
                        print(f"复制失败 {new_filename}: {str(e)}")
            else:
                print(f"警告: 未找到Viking源文件来创建 {unit_name}")
            continue

        # 查找匹配的现有文件
        current_file = find_file(unit_name, race)

        if current_file:
            source_file = os.path.join(source_path, current_file)
            target_file = os.path.join(target_path, new_filename)

            if dry_run:
                print(f"将复制并重命名: {current_file} -> {new_filename}")
            else:
                try:
                    shutil.copy2(source_file, target_file)
                    print(f"成功复制并重命名: {current_file} -> {new_filename}")
                except Exception as e:
                    print(f"复制失败 {current_file}: {str(e)}")
        else:
            print(f"警告: 未找到匹配的文件给 {unit_name}")


if __name__ == "__main__":
    # 设置源文件夹和目标文件夹路径
    source_path = r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info_processed"
    target_path = os.path.join(os.path.dirname(source_path), "sc2_unit_info_processed_vlm_attention")

    # PySC2标准单位名称
    standard_names = [
        'Archon', 'Banshee', 'Ghost', 'Hellbat', 'Immortal',
        'Marauder', 'Marine', 'Medivac', 'Phoenix', 'Reaper',
        'Stalker', 'Viking Assault', 'Zealot', 'Viking Fighter'
    ]

    # 首先运行预览模式
    print("源目录中的文件列表:")
    print("-" * 50)
    for file in os.listdir(source_path):
        print(file)

    print("\n预览模式运行中...\n")
    standardize_unit_files(source_path, target_path, standard_names, dry_run=True)

    # 询问是否继续
    response = input("\n要执行这些更改吗? (yes/no): ")
    if response.lower() == 'yes':
        print("\n执行复制和重命名操作...\n")
        standardize_unit_files(source_path, target_path, standard_names, dry_run=False)
        print("\n操作完成!")
        print(f"\n所有文件已处理并保存到: {target_path}")
        print("\n提示: 请记得在新目录上运行yaml生成脚本来更新配置文件")
    else:
        print("\n操作已取消")