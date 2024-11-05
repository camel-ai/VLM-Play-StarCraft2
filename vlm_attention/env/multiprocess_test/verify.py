import json
# 玩家类型常量
_PLAYER_SELF = 1
_PLAYER_ENEMY = 4

UNIT_ALIGN_DICT = {
    "4294967297": "Marine",
    "4296278017": "Medivac",
    "4298113025": "Marauder",
    "4297588737": "Viking Assault",
    "4297326593": "Ghost",
    "4297064449": "Reaper",
    "4296015873": "Hellbat",
    "4297850881": "Banshee",
    "4295229441": "Archon",
    "4295491585": "Zealot",
    "4296540161": "Immortal",
    "4296802305": "Phoenix",
    "4295753729": "Stalker"
}
PREDEFINED_TAGS = {
    # 我方单位 (alliance = 1)
    (4295229441, _PLAYER_SELF): 1,  # Archon
    (4295491585, _PLAYER_SELF): 2,  # Zealot
    (4296540161, _PLAYER_SELF): 3,  # Immortal
    (4296802305, _PLAYER_SELF): 4,  # Phoenix
    (4295753729, _PLAYER_SELF): 5,  # Stalker

    # 敌方单位 (alliance = 4)
    (4294967297, _PLAYER_ENEMY): 6,  # Marine
    (4296278017, _PLAYER_ENEMY): 7,  # Medivac
    (4298113025, _PLAYER_ENEMY): 8,  # Marauder
    (4297588737, _PLAYER_ENEMY): 9,  # Viking Assault
    (4297326593, _PLAYER_ENEMY): 10,  # Ghost
    (4297064449, _PLAYER_ENEMY): 11,  # Reaper
    (4296015873, _PLAYER_ENEMY): 12,  # Hellbat
    (4297850881, _PLAYER_ENEMY): 13  # Banshee
}
unit_type_dict= {
    "74": "Stalker",
    "73": "Zealot",
    "77": "Sentry",
    "78": "Phoenix",
    "83": "Immortal",
    "141": "Archon",
    "32": "Siege Tank Sieged",
    "49": "Reaper",
    "50": "Ghost",
    "48": "Marine",
    "51": "Marauder",
    "54": "Medivac",
    "484": "Hellbat",
    "55": "Banshee",
    "34": "Viking Assault"
}
import json


def validate_first_frame(json_path):
    """验证第一帧中的单位数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取第一帧数据
    first_frame = data['frames'][0]
    print(f"验证第 {first_frame['frame_id']} 帧:")

    for unit in first_frame['units']:
        unit_tag = str(unit['unit_tag'])
        alliance = unit['alliance']
        unit_type = str(unit['unit_type'])

        # 获取该unit_tag对应的单位名称
        unit_name = UNIT_ALIGN_DICT.get(unit_tag)

        # 获取该unit_type对应的单位类型名称
        type_name = unit_type_dict.get(unit_type)

        print(f"\n单位信息:")
        print(f"  Unit Tag: {unit_tag}")
        print(f"  Alliance: {alliance}")
        print(f"  Unit Type: {unit_type}")

        # 验证unit_type是否正确映射到单位名称
        if type_name != unit_name:
            print(f"  警告: 单位类型名称不匹配")
            print(f"    根据unit_type得到的名称: {type_name}")
            print(f"    根据unit_tag得到的名称: {unit_name}")
        else:
            print(f"  √ 单位类型验证通过: {type_name}")

        # 验证alliance是否合法
        if alliance not in {_PLAYER_SELF, _PLAYER_ENEMY}:
            print(f"  警告: 非法的alliance值: {alliance}")
        else:
            print(f"  √ Alliance验证通过: {alliance}")

        # 验证unit_tag和alliance的组合是否正确
        tag_alliance_key = (int(unit_tag), alliance)
        if tag_alliance_key in PREDEFINED_TAGS:
            print(f"  √ Unit tag和Alliance组合验证通过")
        else:
            print(f"  警告: 未找到unit_tag和alliance的组合: {tag_alliance_key}")

if __name__ == "__main__":
    json_path = r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\env\multiprocess_test\tag_annotation_test\unit_data.json"  # 替换为实际的json文件路径
    validate_first_frame(json_path)