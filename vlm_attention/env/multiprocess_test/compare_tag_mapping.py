import json
from pathlib import Path
from typing import Dict, List
from pprint import pprint

"""
比较多次运行下tag_mapping是否一致
正常情况下，多次运行下tag_mapping应该是一致的
这使得我们可以依据unit tag 来实现单位的唯一标识

"""
def load_json_file(file_path: str) -> Dict:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_unit_info_for_comparison(tag_mapping: Dict) -> Dict:
    """
    提取用于比较的单位信息
    将每个alliance组内的units按original_tag排序，便于比较
    """
    result = {}
    for alliance_key, alliance_data in tag_mapping.items():
        units = alliance_data['units']
        # 按original_tag排序
        sorted_units = sorted(units, key=lambda x: x['original_tag'])
        result[alliance_key] = {
            'total_units': alliance_data['total_units'],
            'tag_range': alliance_data['tag_range'],
            'units': sorted_units
        }
    return result


def compare_tag_mappings(file_paths: List[str]) -> None:
    """比较多个文件中的tag_mapping内容"""
    print(f"开始比较 {len(file_paths)} 个文件的tag_mapping...")

    # 加载所有文件
    mappings = []
    for file_path in file_paths:
        try:
            data = load_json_file(file_path)
            mappings.append({
                'file': file_path,
                'mapping': extract_unit_info_for_comparison(data['tag_mapping'])
            })
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            return

    # 进行比较
    reference = mappings[0]
    all_identical = True

    print("\n=== 详细比较结果 ===")
    for i, current in enumerate(mappings[1:], 1):
        print(f"\n比较 {Path(reference['file']).name} vs {Path(current['file']).name}:")

        if reference['mapping'] == current['mapping']:
            print("✓ tag_mapping完全相同")
            continue

        all_identical = False
        # 详细比较每个alliance组
        all_alliances = set(reference['mapping'].keys()) | set(current['mapping'].keys())

        for alliance in all_alliances:
            if alliance not in reference['mapping']:
                print(f"× {alliance} 只在第{i + 1}个文件中存在")
                continue
            if alliance not in current['mapping']:
                print(f"× {alliance} 只在参考文件中存在")
                continue

            ref_data = reference['mapping'][alliance]
            cur_data = current['mapping'][alliance]

            # 比较单位数量
            if ref_data['total_units'] != cur_data['total_units']:
                print(f"× {alliance} 单位数量不同: {ref_data['total_units']} vs {cur_data['total_units']}")

            # 比较tag范围
            if ref_data['tag_range'] != cur_data['tag_range']:
                print(f"× {alliance} tag范围不同: {ref_data['tag_range']} vs {cur_data['tag_range']}")

            # 比较每个单位
            for j, (ref_unit, cur_unit) in enumerate(zip(ref_data['units'], cur_data['units'])):
                if ref_unit != cur_unit:
                    print(f"× {alliance} 第{j + 1}个单位不同:")
                    print("  参考文件:", ref_unit)
                    print("  当前文件:", cur_unit)

    print("\n=== 总结 ===")
    if all_identical:
        print("✓ 所有文件的tag_mapping完全一致!")
    else:
        print("× 发现不一致，详见上方比较结果")


def main():
    # 指定要比较的文件
    files_to_compare = [
        r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\env\multiprocess_test\tag_annotation_test\unit_data.json",
        r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\env\multiprocess_test\tag_annotation_test\unit_data_2.json",
        r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\env\multiprocess_test\tag_annotation_test\unit_data_3.json"
    ]

    # 确保所有文件都存在
    missing_files = [f for f in files_to_compare if not Path(f).exists()]
    if missing_files:
        print("以下文件不存在:")
        for f in missing_files:
            print(f"- {f}")
        return

    compare_tag_mappings(files_to_compare)


if __name__ == "__main__":
    main()