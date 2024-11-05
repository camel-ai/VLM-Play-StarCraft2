from pysc2.lib.units import Neutral, Protoss, Terran, Zerg

# 颜色定义 (BGR 格式)
COLORS = {
    "self_color": [0, 255, 0],  # 绿色
    "enemy_color": [255, 255, 0]  # 黄色
}


def format_unit_name(enum_name: str) -> str:
    """
    将枚举名称转换为更易读的格式
    例如: 'SiegeTankSieged' -> 'Siege Tank Sieged'

    Args:
        enum_name: 枚举中的单位名称

    Returns:
        str: 格式化后的单位名称
    """
    # 在大写字母前添加空格，除非是第一个字符
    formatted = ''.join(' ' + c if c.isupper() and i > 0 else c
                        for i, c in enumerate(enum_name))
    return formatted.strip()


def get_unit_name(unit_type: int) -> str:
    """
    通过unit_type获取单位名称
    使用pysc2.lib.units中的枚举定义

    Args:
        unit_type: 单位类型ID

    Returns:
        str: 格式化的单位名称，如果未找到则返回Unknown
    """
    # 按种族顺序尝试获取单位类型
    for race in [Protoss, Terran, Zerg, Neutral]:
        try:
            unit_enum = race(unit_type)
            return format_unit_name(unit_enum.name)
        except ValueError:
            continue

    return f"Unknown Unit {unit_type}"
