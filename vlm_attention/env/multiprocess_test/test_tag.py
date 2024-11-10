import os
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app, flags
import json
import cv2
import numpy as np
from vlm_attention.run_env.utils import _annotate_units_on_image
from collections import defaultdict
from datetime import datetime

FLAGS = flags.FLAGS
map_list = ["vlm_attention_1",
            "2c_vs_64zg_vlm_attention",
            "2m_vs_1z_vlm_attention",
            "2s_vs_1sc_vlm_attention",
            "2s3z_vlm_attention",
            "3m_vlm_attention",
            "3s_vs_3z_vlm_attention"]
# 定义屏幕尺寸常量
FEATURE_SCREEN_SIZE = (256, 256)  # (width, height)
RGB_SCREEN_SIZE = (1920, 1080)  # (width, height)

# 定义颜色常量 (BGR 格式)
COLOR_SELF = (0, 255, 0)  # 绿色 (BGR)
COLOR_ENEMY = (0, 0, 255)  # 红色 (BGR)
COLOR_OTHER = (0, 255, 255)  # 黄色 (BGR)

"""
测试并保存单位信息和图像

单位信息主要包括:
- simple_tag: 简化的tag编号
- alliance: 阵营编号
- unit_type: 单位类型编号
- original_tag: 原始tag编号

图像上的单位标注信息:
- tag_index: simple_tag
- position: 位置坐标 (x, y)
- color: 颜色 (BGR)


"""
class UnitInfo:
    def __init__(self, tag, alliance, unit_type, simplified_tag=None):
        self.tag = int(tag)
        self.alliance = int(alliance)
        self.unit_type = int(unit_type)
        self.simplified_tag = simplified_tag

    def to_dict(self):
        return {
            'original_tag': self.tag,
            'simplified_tag': self.simplified_tag,
            'alliance': self.alliance,
            'unit_type': self.unit_type
        }


class TagAnnotationAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TagAnnotationAgent, self).__init__()
        self.unit_data = []
        self.frame_count = 0
        self.tag_mapping = {}
        self.alliance_groups = defaultdict(list)
        self.next_tag_start = 1
        self.map_info = None

        # 创建日志目录
        self.log_dir = self._create_log_directories()

    def _create_log_directories(self):
        """创建日志目录结构"""
        # 获取当前时间
        current_time = datetime.now()
        # 创建主目录名称 (格式: YYYY-MM-DD_HH-MM-SS)
        dir_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

        # 在当前目录下创建主目录
        log_dir = os.path.join(os.getcwd(), "sc2_logs", dir_name)

        # 创建子目录
        subdirs = {
            'images': os.path.join(log_dir, 'images'),
            'data': os.path.join(log_dir, 'data'),
        }

        # 创建所有必要的目录
        for dir_path in subdirs.values():
            os.makedirs(dir_path, exist_ok=True)

        return {
            'base_dir': log_dir,
            'images_dir': subdirs['images'],
            'data_dir': subdirs['data']
        }

    def save_image(self, image, filename):
        """保存图像到指定目录"""
        full_path = os.path.join(self.log_dir['images_dir'], filename)
        cv2.imwrite(full_path, image)

    def save_json_data(self, data, filename):
        """保存JSON数据到指定目录"""
        full_path = os.path.join(self.log_dir['data_dir'], filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_map_info(self, obs):
        """获取地图信息"""
        map_info = {}

        # 使用feature_minimap获取真实地图尺寸
        if 'feature_minimap' in obs.observation:
            minimap_layers = obs.observation['feature_minimap']
            # 从heightmap获取真实地图尺寸
            height_map = minimap_layers[0]
            map_info['real_map_size'] = {
                'width': int(height_map.shape[1]),
                'height': int(height_map.shape[0])
            }

            # 获取可玩区域信息
            visibility_map = minimap_layers[1]
            playable_areas = np.where(visibility_map > 0)
            if len(playable_areas[0]) > 0 and len(playable_areas[1]) > 0:
                min_y, max_y = int(np.min(playable_areas[0])), int(np.max(playable_areas[0]))
                min_x, max_x = int(np.min(playable_areas[1])), int(np.max(playable_areas[1]))
                map_info['playable_area'] = {
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y,
                    'width': max_x - min_x + 1,
                    'height': max_y - min_y + 1
                }

            # 获取地形高度信息
            height_map = obs.observation['feature_minimap'][0]
            map_info['terrain_height'] = {
                'min': float(np.min(height_map)),
                'max': float(np.max(height_map))
            }

        # 获取地图名称
        if 'map_name' in obs.observation:
            map_info['map_name'] = str(obs.observation['map_name'])

        return map_info
    def save_episode_summary(self):
        """保存回合总结数据"""
        summary_data = {
            'feature_screen_size': list(FEATURE_SCREEN_SIZE),
            'rgb_screen_size': list(RGB_SCREEN_SIZE),
            'map_info': self.map_info,
            'tag_mapping': self.get_grouped_tag_mapping(),
            'total_frames': self.frame_count,
            'frames': self.unit_data
        }

        # 保存总结数据
        self.save_json_data(summary_data, 'episode_summary.json')

        # 保存基本信息到单独的文件
        basic_info = {
            'feature_screen_size': list(FEATURE_SCREEN_SIZE),
            'rgb_screen_size': list(RGB_SCREEN_SIZE),
            'map_info': self.map_info,
            'total_frames': self.frame_count
        }
        self.save_json_data(basic_info, 'basic_info.json')

        # 保存标签映射到单独的文件
        self.save_json_data(self.get_grouped_tag_mapping(), 'tag_mapping.json')

    def get_grouped_tag_mapping(self):
        """生成按alliance分组的tag_mapping信息"""
        grouped_mapping = defaultdict(list)

        # 首先确保所有单位都有simplified_tag
        for tag, unit_info in self.tag_mapping.items():
            # 如果发现任何没有simplified_tag的单位，重新初始化
            if unit_info.simplified_tag is None:
                self.initialize_tag_mapping()
                break

        # 按alliance分组收集信息
        for tag, unit_info in self.tag_mapping.items():
            if unit_info.simplified_tag is not None:  # 添加安全检查
                grouped_mapping[unit_info.alliance].append({
                    'original_tag': unit_info.tag,
                    'simplified_tag': unit_info.simplified_tag,
                    'unit_type': unit_info.unit_type
                })

        # 对每个组内的单位按simplified_tag排序 (添加错误处理)
        for alliance in grouped_mapping:
            try:
                grouped_mapping[alliance].sort(key=lambda x: x['simplified_tag'] or float('inf'))
            except TypeError as e:
                print(f"Warning: sorting error for alliance {alliance}: {e}")
                print(f"Problem units: {grouped_mapping[alliance]}")
                continue

        # 转换为最终输出格式
        return {
            f"alliance_{alliance}": {
                "total_units": len(units),
                "tag_range": f"{units[0]['simplified_tag'] if units else 'N/A'}-{units[-1]['simplified_tag'] if units else 'N/A'}",
                "units": units
            }
            for alliance, units in grouped_mapping.items()
        }

    def update_alliance_groups(self, units):
        """更新阵营分组信息"""
        # 清空现有分组
        temp_groups = defaultdict(list)
        needs_initialization = False

        # 收集新的单位信息
        for unit in units:
            original_tag = int(unit.tag)
            alliance = int(unit.alliance)
            unit_type = int(unit.unit_type)

            # 更新或创建单位信息
            if original_tag not in self.tag_mapping:
                self.tag_mapping[original_tag] = UnitInfo(
                    tag=original_tag,
                    alliance=alliance,
                    unit_type=unit_type
                )
                needs_initialization = True

            temp_groups[alliance].append(original_tag)

        # 对每个阵营组内的tag排序
        for alliance in temp_groups:
            temp_groups[alliance].sort(key=lambda t: (
                self.tag_mapping[t].unit_type,
                t
            ))

        # 更新alliance_groups
        self.alliance_groups = temp_groups

        # 如果有新单位或发现任何没有simplified_tag的单位，重新初始化
        if needs_initialization or any(unit.simplified_tag is None for unit in self.tag_mapping.values()):
            self.initialize_tag_mapping()

    def initialize_tag_mapping(self):
        """初始化简化tag映射关系"""
        current_tag = 1
        print("Initializing tag mapping...")

        # 调整排序规则，使其更稳定
        alliance_order = sorted(self.alliance_groups.keys())

        for alliance in alliance_order:
            # 获取该alliance下的所有单位信息并排序
            units = [(tag, self.tag_mapping[tag]) for tag in self.alliance_groups[alliance]]
            units.sort(key=lambda x: (x[1].unit_type, x[0]))

            for original_tag, _ in units:
                self.tag_mapping[original_tag].simplified_tag = current_tag
                print(f"Assigned simplified_tag {current_tag} to original_tag {original_tag}")
                current_tag += 1

    def get_simplified_tag(self, original_tag):
        """获取简化的tag编号"""
        original_tag = int(original_tag)
        if original_tag in self.tag_mapping:
            return self.tag_mapping[original_tag].simplified_tag
        return -1
    def convert_screen_to_rgb_coords(self, x, y):
        """转换坐标从特征层到RGB层"""
        rgb_x = int((x / FEATURE_SCREEN_SIZE[0]) * RGB_SCREEN_SIZE[0])
        rgb_y = int((y / FEATURE_SCREEN_SIZE[1]) * RGB_SCREEN_SIZE[1])
        return rgb_x, rgb_y
    def step(self, obs):
        super(TagAnnotationAgent, self).step(obs)

        # 在第一帧获取地图信息
        if self.frame_count == 0:
            self.map_info = self.get_map_info(obs)
            # 保存地图信息
            self.save_json_data(self.map_info, 'map_info.json')

        if 'rgb_screen' in obs.observation:
            # 更新阵营分组信息
            self.update_alliance_groups(obs.observation.feature_units)

            # 获取图像并准备标注
            screen = np.array(obs.observation['rgb_screen'], dtype=np.uint8)
            units_to_annotate = []
            for unit in obs.observation.feature_units:
                simple_tag = self.get_simplified_tag(unit.tag)
                rgb_x, rgb_y = self.convert_screen_to_rgb_coords(unit.x, unit.y)

                color = COLOR_SELF if unit.alliance == 1 else COLOR_ENEMY if unit.alliance == 4 else COLOR_OTHER
                unit_info = {
                    'tag_index': simple_tag,
                    'position': (rgb_x, rgb_y),
                    'color': color
                }
                units_to_annotate.append(unit_info)

            try:
                # 在图像上添加标注
                annotated_image = _annotate_units_on_image(
                    screen,
                    units_to_annotate,
                    circle_radius=15,
                    font_scale=0.8,
                    circle_thickness=2
                )

                # 保存图像
                self.save_image(annotated_image, f'frame_{self.frame_count:04d}.png')

            except Exception as e:
                print(f"处理图像时出错: {str(e)}")

            # 存储单位数据
            current_frame = {
                'game_loop': int(obs.observation.game_loop[0]),
                'frame_id': self.frame_count,
                'units': [{
                    'simple_tag': self.get_simplified_tag(unit.tag),
                    'alliance': int(unit.alliance),
                    'unit_type': int(unit.unit_type),
                    'original_tag': int(unit.tag),
                } for unit in obs.observation.feature_units],
                'alliance_groups': {
                    str(alliance): [int(tag) for tag in tags]
                    for alliance, tags in self.alliance_groups.items()
                }
            }

            # 保存当前帧数据
            self.save_json_data(current_frame, f'frame_{self.frame_count:04d}.json')

            self.unit_data.append(current_frame)
            self.frame_count += 1

        return actions.RAW_FUNCTIONS.no_op()


def run_test():
    agent = TagAnnotationAgent()

    try:
        with sc2_env.SC2Env(
                map_name=map_list[2],
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=FEATURE_SCREEN_SIZE, minimap=64),
                    rgb_dimensions=features.Dimensions(screen=RGB_SCREEN_SIZE, minimap=(64, 64)),
                    action_space=actions.ActionSpace.RAW,
                    use_feature_units=True,
                    use_raw_units=True,
                    use_unit_counts=True,
                    use_camera_position=True,
                    show_cloaked=True,
                    show_burrowed_shadows=True,
                    show_placeholders=True,
                    raw_crop_to_playable_area=True,
                    raw_resolution=64
                ),
                step_mul=8,
                game_steps_per_episode=1000,
                visualize=True) as env:

            agent.setup(env.observation_spec(), env.action_spec())
            obs = env.reset()
            agent.reset()

            while True:
                step_actions = [agent.step(obs[0])]

                if obs[0].last():
                    # 保存回合总结数据
                    agent.save_episode_summary()
                    print(f"\n测试完成, 共处理 {agent.frame_count} 帧")
                    print(f"日志保存在目录: {agent.log_dir['base_dir']}")
                    break

                obs = env.step(step_actions)

    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        raise


def main(argv):
    flags.FLAGS(argv)
    run_test()


if __name__ == "__main__":
    app.run(main)