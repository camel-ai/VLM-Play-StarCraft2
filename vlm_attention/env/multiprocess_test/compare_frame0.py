from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import json
from pathlib import Path


"""
unit class for starcraft 2

this class is used to store the unit information

it is used in the frame class
"""

@dataclass
class Unit:
    simple_tag: int
    alliance: int
    unit_type: int
    original_tag: int

    @classmethod
    def from_dict(cls, data: dict) -> 'Unit':
        return cls(
            simple_tag=data['simple_tag'],
            alliance=data['alliance'],
            unit_type=data['unit_type'],
            original_tag=data['original_tag']
        )


@dataclass
class Frame:
    game_loop: int
    frame_id: int
    units: List[Unit]
    alliance_groups: Dict[str, List[int]]

    @classmethod
    def from_json(cls, data: dict) -> 'Frame':
        return cls(
            game_loop=data['game_loop'],
            frame_id=data['frame_id'],
            units=[Unit.from_dict(u) for u in data['units']],
            alliance_groups=data['alliance_groups']
        )


class FrameComparator:
    def __init__(self, logs_dir: str = "sc2_logs"):
        self.logs_dir = Path(logs_dir)
        self.results = {
            'total_logs': 0,
            'consistent': True,
            'differences': [],
            'log_details': [],
            'frame1_contents': {}
        }

    def find_log_directories(self) -> List[Path]:
        """查找所有有效的日志目录"""
        log_dirs = []
        for entry in self.logs_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                datetime.strptime(entry.name, "%Y-%m-%d_%H-%M-%S")
                log_dirs.append(entry)
            except ValueError:
                continue
        return sorted(log_dirs)

    def load_frame(self, log_dir: Path) -> Optional[Frame]:
        """从日志目录加载frame_0000.json"""
        frame_path = log_dir / "data" / "frame_0001.json"
        if not frame_path.exists():
            print(f"警告: {log_dir} 中未找到 frame_0001.json")
            return None

        try:
            with frame_path.open('r', encoding='utf-8') as f:
                return Frame.from_json(json.load(f))
        except Exception as e:
            print(f"错误: 处理 {frame_path} 时发生异常: {str(e)}")
            return None

    def compare_units(self, ref_units: List[Unit], curr_units: List[Unit]) -> dict:
        """比较两组单位的差异"""
        differences = []

        for i in range(max(len(ref_units), len(curr_units))):
            if i >= len(ref_units):
                differences.append({
                    'type': 'extra_unit',
                    'index': i,
                    'current_unit': curr_units[i].__dict__
                })
            elif i >= len(curr_units):
                differences.append({
                    'type': 'missing_unit',
                    'index': i,
                    'reference_unit': ref_units[i].__dict__
                })
            elif ref_units[i].__dict__ != curr_units[i].__dict__:
                diff_fields = {}
                for field in ['simple_tag', 'alliance', 'unit_type', 'original_tag']:
                    ref_value = getattr(ref_units[i], field)
                    curr_value = getattr(curr_units[i], field)
                    if ref_value != curr_value:
                        diff_fields[field] = {
                            'reference': ref_value,
                            'current': curr_value
                        }
                if diff_fields:
                    differences.append({
                        'type': 'mismatch',
                        'index': i,
                        'fields': diff_fields
                    })

        return {
            'total_units': {
                'reference': len(ref_units),
                'current': len(curr_units)
            },
            'differences': differences
        } if differences else {}

    def compare_alliance_groups(self, ref_groups: Dict, curr_groups: Dict) -> dict:
        """比较联盟组的差异"""
        differences = {}
        all_alliances = set(ref_groups.keys()) | set(curr_groups.keys())

        for alliance in all_alliances:
            ref_tags = set(ref_groups.get(alliance, []))
            curr_tags = set(curr_groups.get(alliance, []))

            if ref_tags != curr_tags:
                differences[alliance] = {
                    'missing_tags': sorted(list(ref_tags - curr_tags)),
                    'extra_tags': sorted(list(curr_tags - ref_tags))
                }

        return differences

    def analyze_differences(self, ref_frame: Frame, curr_frame: Frame) -> dict:
        """分析两个Frame之间的所有差异"""
        differences = {}

        # 比较单位
        unit_differences = self.compare_units(ref_frame.units, curr_frame.units)
        if unit_differences:
            differences['units'] = unit_differences

        # 比较联盟组
        alliance_differences = self.compare_alliance_groups(
            ref_frame.alliance_groups,
            curr_frame.alliance_groups
        )
        if alliance_differences:
            differences['alliance_groups'] = alliance_differences

        if differences:
            differences['summary'] = {
                'has_unit_differences': bool(unit_differences),
                'has_alliance_differences': bool(alliance_differences),
                'reference_units_count': len(ref_frame.units),
                'current_units_count': len(curr_frame.units)
            }

        return differences

    def compare_frames(self) -> Tuple[bool, dict]:
        """执行帧比较的主函数"""
        log_dirs = self.find_log_directories()
        if not log_dirs:
            print("未找到任何日志目录!")
            return False, self.results

        self.results['total_logs'] = len(log_dirs)
        print(f"找到 {len(log_dirs)} 个日志目录")

        # 获取参考帧
        reference_dir = log_dirs[0]
        reference_frame = self.load_frame(reference_dir)
        if not reference_frame:
            return False, self.results

        # 比较其他帧
        for log_dir in log_dirs[1:]:
            current_frame = self.load_frame(log_dir)
            if not current_frame:
                continue

            # 记录日志详情
            self.results['log_details'].append({
                'dir': log_dir.name,
                'timestamp': log_dir.name,
                'units_count': len(current_frame.units),
                'alliances': list(current_frame.alliance_groups.keys())
            })

            # 比较并记录差异
            differences = self.analyze_differences(reference_frame, current_frame)
            if differences:
                self.results['consistent'] = False
                self.results['differences'].append({
                    'dir': log_dir.name,
                    'reference_dir': reference_dir.name,
                    'differences': differences
                })

        return self.results['consistent'], self.results

    def save_results(self):
        """保存比较结果"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"frame1_comparison_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n比较结果已保存到: {filename}")
        except Exception as e:
            print(f"保存比较结果时出错: {str(e)}")


def main():
    comparator = FrameComparator()
    is_consistent, results = comparator.compare_frames()
    comparator.save_results()

    # 输出总结
    print("\n=== 比较结果总结 ===")
    print(f"总共比较了 {results['total_logs']} 个日志目录")
    print(f"数据一致性: {'一致' if is_consistent else '不一致'}")
    if not is_consistent:
        print(f"发现 {len(results['differences'])} 处差异")


if __name__ == "__main__":
    main()