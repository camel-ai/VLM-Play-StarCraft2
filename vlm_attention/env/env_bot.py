import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions, features
from collections import defaultdict
from vlm_attention.env.config import COLORS, get_unit_name

import logging

# 设置logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
"""
environment bot without ability support

this bot provide an easier interface for building the environment without ability support

based on this bot, environment can directly interact with agent through out text and image

"""
# 玩家类型常量
_PLAYER_SELF = 1
_PLAYER_ENEMY = 4

# 方向常量
UP, RIGHT, DOWN, LEFT = range(4)


class UnitInfo:
    def __init__(self, unit, alliance, simplified_tag, feature_dims, rgb_dims):
        if feature_dims is None or rgb_dims is None:
            raise ValueError("feature_dims and rgb_dims must be provided to UnitInfo")

        self.original_tag = int(unit.tag)
        self.alliance = int(alliance)
        self.unit_type = int(unit.unit_type)
        self.simplified_tag = simplified_tag
        self.max_health = float(unit.health)
        self.max_shield = float(unit.shield)
        self.alive = True
        self.feature_dims = feature_dims
        self.rgb_dims = rgb_dims
        self.logger = logging.getLogger('UnitInfo')
        self.update_status(unit)

    def update_status(self, unit):
        """更新单位状态"""
        self.health = float(unit.health)
        self.shield = float(unit.shield)
        self.energy = float(unit.energy)
        self.x = float(unit.x)
        self.y = float(unit.y)

        try:
            # 计算网格位置 - 使用feature_dims来计算网格位置
            self.grid_x = int(self.x * 10 / self.feature_dims[0])
            self.grid_y = int(self.y * 10 / self.feature_dims[1])
            # 确保网格坐标在有效范围内
            self.grid_x = max(0, min(self.grid_x, 9))
            self.grid_y = max(0, min(self.grid_y, 9))
            self.logger.debug(f"Calculated grid position: ({self.grid_x}, {self.grid_y}) "
                              f"from screen position: ({self.x}, {self.y})")
        except Exception as e:
            self.logger.error(f"Error calculating grid position: {e}")
            self.grid_x = 0
            self.grid_y = 0

        self.alive = self.health > 0

    def to_dict(self):
        """转换为字典格式"""
        try:
            return {
                'original_tag': self.original_tag,
                'simplified_tag': self.simplified_tag,
                'alliance': self.alliance,
                'unit_type': self.unit_type,
                'health': self.health,
                'max_health': self.max_health,
                'shield': self.shield,
                'max_shield': self.max_shield,
                'energy': self.energy,
                'position': (self.x, self.y),
                'grid_position': (self.grid_x, self.grid_y)
            }
        except Exception as e:
            self.logger.error(f"Error in to_dict: {e}")
            # 返回一个带有默认值的字典
            return {
                'original_tag': self.original_tag,
                'simplified_tag': self.simplified_tag,
                'alliance': self.alliance,
                'unit_type': self.unit_type,
                'health': 0.0,
                'max_health': 0.0,
                'shield': 0.0,
                'max_shield': 0.0,
                'energy': 0.0,
                'position': (0.0, 0.0),
                'grid_position': (0, 0)
            }


class UnitManager:
    def __init__(self, feature_dims, rgb_dims):
        self.unit_info = {}
        self.alliance_groups = defaultdict(list)
        self.tag_registry = {}
        self.type_counters = {}
        self.next_simplified_tag = 1
        self.initialized = False
        # 确保存储尺寸信息
        self.feature_dims = feature_dims
        self.rgb_dims = rgb_dims

        # 添加日志记录
        self.logger = logging.getLogger('UnitManager')
        self.logger.info(f"Initialized UnitManager with feature_dims={feature_dims}, rgb_dims={rgb_dims}")

    def initialize_units(self, units):
        """初始化单位注册信息，只在第一次调用时执行"""
        if self.initialized:
            return

        # 按阵营和单位类型对单位进行排序
        sorted_units = []
        for unit in units:
            if unit.alliance in (_PLAYER_SELF, _PLAYER_ENEMY):
                sorted_units.append(unit)

        sorted_units.sort(key=lambda x: (
            x.alliance != _PLAYER_SELF,  # 己方单位优先
            x.unit_type,  # 按单位类型
            x.x,  # 按x坐标
            x.y,  # 按y坐标
            x.tag  # 最后按原始tag
        ))

        # 为每个单位分配永久性标识
        for unit in sorted_units:
            original_tag = int(unit.tag)
            unit_type = int(unit.unit_type)
            alliance = int(unit.alliance)

            # 获取该类型单位的计数
            type_key = (unit_type, alliance)
            if type_key not in self.type_counters:
                self.type_counters[type_key] = 1
            else:
                self.type_counters[type_key] += 1

            # 生成单位名称
            base_name = get_unit_name(unit_type)
            unit_name = f"{base_name}_{self.type_counters[type_key]}"

            # 注册单位信息
            self.tag_registry[original_tag] = (self.next_simplified_tag, unit_name)
            self.next_simplified_tag += 1

            # 记录日志
            print(f"Registered unit: {unit_name} (original_tag: {original_tag}, "
                  f"simplified_tag: {self.next_simplified_tag - 1})")

        self.initialized = True

    def get_simplified_tag(self, original_tag):
        """获取单位的simplified tag

        Args:
            original_tag: 原始tag

        Returns:
            int: simplified tag，如果不存在则返回-1
        """
        unit_info = self.tag_registry.get(int(original_tag))
        if unit_info:
            return unit_info[0]  # 返回(simplified_tag, unit_name)中的simplified_tag
        return -1

    def update_units(self, units):
        """更新单位状态，保持标识符的一致性"""
        if not self.initialized:
            self.initialize_units(units)

        self.alliance_groups.clear()

        for unit in units:
            original_tag = int(unit.tag)
            alliance = int(unit.alliance)

            if alliance not in (_PLAYER_SELF, _PLAYER_ENEMY):
                continue

            unit_info = self.tag_registry.get(original_tag)
            if unit_info is None:
                unit_type = int(unit.unit_type)
                type_key = (unit_type, alliance)

                if type_key not in self.type_counters:
                    self.type_counters[type_key] = 1
                else:
                    self.type_counters[type_key] += 1

                base_name = get_unit_name(unit_type)
                unit_name = f"{base_name}_{self.type_counters[type_key]}"

                self.tag_registry[original_tag] = (self.next_simplified_tag, unit_name)
                unit_info = (self.next_simplified_tag, unit_name)
                self.next_simplified_tag += 1

                self.logger.info(f"New unit registered: {unit_name} (original_tag: {original_tag}, "
                                 f"simplified_tag: {unit_info[0]})")

            # 更新或创建UnitInfo对象，确保传入尺寸参数
            if original_tag not in self.unit_info:
                self.unit_info[original_tag] = UnitInfo(unit, alliance, unit_info[0],
                                                        self.feature_dims, self.rgb_dims)
            else:
                self.unit_info[original_tag].update_status(unit)

            self.alliance_groups[alliance].append(original_tag)

        # 对alliance组内的单位按simplified_tag排序
        for alliance in self.alliance_groups:
            self.alliance_groups[alliance].sort(
                key=lambda x: self.tag_registry[x][0]
            )

    def get_original_tag_by_simplified(self, simplified_tag):
        """根据simplified_tag获取original_tag"""
        try:
            # 从tag_registry中查找
            for original_tag, (reg_simplified_tag, _) in self.tag_registry.items():
                if reg_simplified_tag == simplified_tag:
                    # 验证单位是否存活
                    if original_tag in self.unit_info and self.unit_info[original_tag].alive:
                        return original_tag
            return None
        except Exception as e:
            logging.error(f"Error getting original tag: {e}")
            return None

    def is_valid_unit(self, original_tag):
        """检查单位是否有效"""
        return (original_tag in self.unit_info and
                self.unit_info[original_tag].alive)


class Multimodal_bot(base_agent.BaseAgent):
    """StarCraft II 多模态机器人代理

    该代理能够:
    1. 处理和生成多模态观察（图像、文本描述和单位信息）
    2. 执行多种类型的动作（攻击、移动）
    3. 管理单位信息和状态

    Attributes:
        unit_manager (UnitManager): 管理游戏中单位的信息和状态
        step_count (int): 当前步数
        self_color (tuple): 己方单位的RGB颜色
        enemy_color (tuple): 敌方单位的RGB颜色
        feature_screen_size (tuple): 特征屏幕尺寸
        image_size (tuple): 图像尺寸
        game_map_size (tuple): 游戏地图尺寸
        attack_commands (list): 当前回合的攻击命令列表
        original_move_commands (list): 当前回合的移动命令列表
        smac_move_commands (list): 当前回合的SMAC风格移动命令列表
        max_health_shield (dict): 记录单位的最大生命值和护盾
    """

    def __init__(self, self_color=(0, 255, 0), enemy_color=(0, 0, 255), feature_dims=None, rgb_dims=None,
                 map_size=None):
        super(Multimodal_bot, self).__init__()

        # 添加参数验证
        if feature_dims is None or rgb_dims is None:
            raise ValueError("feature_dims and rgb_dims must be provided")

        self.logger = logging.getLogger('Multimodal_bot')
        self.logger.info(f"Initializing Multimodal_bot with feature_dims={feature_dims}, rgb_dims={rgb_dims}")

        self.unit_manager = UnitManager(feature_dims, rgb_dims)
        self.step_count = 0
        self.self_color = self_color
        self.enemy_color = enemy_color
        self.feature_screen_size = feature_dims
        self.image_size = rgb_dims
        self.game_map_size = map_size

        # 添加命令存储
        self.attack_commands = []
        self.original_move_commands = []
        self.smac_move_commands = []
        self.max_health_shield = {}

        # 添加动作缓存
        self._move_cache = {}
        self._attack_cache = {}
        self._coordinate_cache = {}

        # 添加错误计数器
        self._error_counts = defaultdict(int)
        self.max_errors = 3  # 最大错误次数

        # 初始化缓存和计数器
        self._move_cache = {}
        self._attack_cache = {}
        self._coordinate_cache = {}
        self._error_counts = defaultdict(int)
        self._simplified_to_original_cache = {}  # 添加这个缓存的初始化
        self.max_errors = 3

    def step(self, obs):
        """处理每一步的观察并返回动作列表"""
        self.logger.info(f"\n=== Step {self.step_count} ===")

        # 第一步时记录单位的最大生命值和护盾
        if self.step_count == 1:
            for unit in obs.observation.raw_units:
                self.max_health_shield[unit.tag] = (unit.health, unit.shield)

        # 调用父类的step方法
        super(Multimodal_bot, self).step(obs)

        # 更新单位管理器
        self.unit_manager.update_units(obs.observation.raw_units)

        actions_list = []
        executed_actions = {
            'attack': [],
            'move': [],
            'failed': []
        }

        # 验证并执行攻击命令
        for attacker_tag, target_tag in self.attack_commands:
            is_valid, error_msg = self.validate_attack_command(attacker_tag, target_tag)

            if not is_valid:
                executed_actions['failed'].append({
                    'type': 'attack',
                    'attacker': attacker_tag,
                    'target': target_tag,
                    'reason': error_msg
                })
                continue

            # 获取原始tag（已经在validate_attack_command中验证过了）
            attacker_original = self.unit_manager.get_original_tag_by_simplified(attacker_tag)
            target_original = self.unit_manager.get_original_tag_by_simplified(target_tag)

            # 创建攻击动作
            action = actions.RAW_FUNCTIONS.Attack_unit("now", attacker_original, target_original)
            if action:
                actions_list.append(action)
                executed_actions['attack'].append({
                    'attacker': attacker_tag,
                    'target': target_tag,
                    'action': str(action)
                })
            else:
                executed_actions['failed'].append({
                    'type': 'attack',
                    'attacker': attacker_tag,
                    'target': target_tag,
                    'reason': 'Failed to create action'
                })

        # 处理原始移动命令
        for unit_simplified_tag, grid_pos in self.original_move_commands:
            action = self.create_move_action(unit_simplified_tag, grid_pos, obs)
            if action is not None:
                actions_list.append(action)
                executed_actions['move'].append({
                    'unit': unit_simplified_tag,
                    'target': grid_pos,
                    'type': 'grid',
                    'action': str(action)
                })
            else:
                executed_actions['failed'].append({
                    'type': 'grid_move',
                    'unit': unit_simplified_tag,
                    'target': grid_pos
                })

        # 处理SMAC移动命令
        for unit_simplified_tag, direction in self.smac_move_commands:
            action = self.create_smac_move_action(unit_simplified_tag, direction, obs)
            if action is not None:
                actions_list.append(action)
                direction_name = ["UP", "RIGHT", "DOWN", "LEFT"][direction]
                executed_actions['move'].append({
                    'unit': unit_simplified_tag,
                    'direction': direction_name,
                    'type': 'smac',
                    'action': str(action)
                })
            else:
                executed_actions['failed'].append({
                    'type': 'smac_move',
                    'unit': unit_simplified_tag,
                    'direction': direction
                })

        # 详细的日志记录
        self.logger.info("\nAction Execution Summary:")
        self.logger.info(f"Successfully executed attacks: {len(executed_actions['attack'])}")
        for attack in executed_actions['attack']:
            self.logger.info(f"Attack: Unit {attack['attacker']} -> Unit {attack['target']}")
            self.logger.info(f"Raw action: {attack['action']}")

        self.logger.info(f"\nSuccessfully executed moves: {len(executed_actions['move'])}")
        for move in executed_actions['move']:
            if move['type'] == 'grid':
                self.logger.info(f"Grid Move: Unit {move['unit']} -> Position {move['target']}")
            else:
                self.logger.info(f"SMAC Move: Unit {move['unit']} -> {move['direction']}")
            self.logger.info(f"Raw action: {move['action']}")

        if executed_actions['failed']:
            self.logger.warning("\nFailed actions:")
            for failed in executed_actions['failed']:
                self.logger.warning(f"Failed {failed['type']}: {failed}")

        # 如果没有任何命令，执行空操作
        if not actions_list:
            actions_list.append(actions.RAW_FUNCTIONS.no_op())
            self.logger.info("\nNo action - executing no_op")
        # 只记录非空操作的动作
        elif not (len(actions_list) == 1 and actions_list[0].function == actions.RAW_FUNCTIONS.no_op):
            self.logger.info(f"Step {self.step_count} actions: {actions_list}")
        # 清除本轮的命令
        self.attack_commands.clear()
        self.original_move_commands.clear()
        self.smac_move_commands.clear()

        self.step_count += 1
        return actions_list

    def reset(self):
        """重置智能体状态"""
        super(Multimodal_bot, self).reset()
        self.step_count = 0
        self.attack_commands.clear()
        self.original_move_commands.clear()
        self.smac_move_commands.clear()
        self.max_health_shield.clear()

        # 清理所有缓存
        self._move_cache.clear()
        self._attack_cache.clear()
        self._coordinate_cache.clear()
        self._error_counts.clear()
        self._simplified_to_original_cache.clear()

    def add_attack_command(self, attacker_simplified_tag, target_simplified_tag):
        """添加攻击命令

        Args:
            attacker_simplified_tag (int): 攻击者的简化ID
            target_simplified_tag (int): 目标的简化ID
        """
        # 验证单位是否存在且有效
        attacker_tag = self.get_original_tag(attacker_simplified_tag)
        target_tag = self.get_original_tag(target_simplified_tag)

        if attacker_tag is None or target_tag is None:
            return  # 无效的单位ID

        attacker_info = self.unit_manager.unit_info[attacker_tag]
        target_info = self.unit_manager.unit_info[target_tag]

        # 验证攻击者是己方单位，目标是敌方单位
        if (attacker_info.alliance == _PLAYER_SELF and
                target_info.alliance == _PLAYER_ENEMY and
                attacker_info.health > 0):  # 确保攻击者存活
            self.attack_commands.append((attacker_simplified_tag, target_simplified_tag))

    def add_move_command(self, unit_simplified_tag, grid_pos):
        """添加移动命令"""
        self.original_move_commands.append((unit_simplified_tag, grid_pos))

    def add_smac_move_command(self, unit_simplified_tag, direction):
        """添加SMAC移动命令"""
        self.smac_move_commands.append((unit_simplified_tag, direction))

    def get_raw_image_and_unit_info(self, obs):
        """获取原始图像和详细的单位信息"""
        # 更新单位信息
        self.unit_manager.update_units(obs.observation.raw_units)

        # 获取RGB屏幕图像
        if 'rgb_screen' in obs.observation:
            frame = np.array(obs.observation['rgb_screen'], dtype=np.uint8)
        else:
            frame = np.zeros((*self.image_size, 3), dtype=np.uint8)

        # 收集单位信息
        unit_info = []
        for unit in obs.observation.raw_units:
            if unit.alliance in (_PLAYER_SELF, _PLAYER_ENEMY):
                original_tag = int(unit.tag)
                unit_registry = self.unit_manager.tag_registry.get(original_tag)

                if unit_registry:
                    simplified_tag = unit_registry[0]  # 获取simplified_tag
                    for feature_unit in obs.observation.feature_units:
                        if feature_unit.tag == unit.tag:
                            # 转换坐标
                            screen_x = int(feature_unit.x * self.image_size[0] / self.feature_screen_size[0])
                            screen_y = int(feature_unit.y * self.image_size[1] / self.feature_screen_size[1])

                            color = self.self_color if unit.alliance == _PLAYER_SELF else self.enemy_color

                            # 获取单位详细信息
                            unit_data = self.unit_manager.unit_info[original_tag].to_dict()
                            unit_data.update({
                                'position': (screen_x, screen_y),
                                'map_position': (unit.x, unit.y),
                                'color': color
                            })
                            unit_info.append(unit_data)
                            break

        return frame, unit_info

    def get_text_description(self, obs):
        """获取游戏状态的文字描述"""
        description = "Current game state:\n\n"

        # 按alliance分组输出单位信息
        for alliance in sorted(self.unit_manager.alliance_groups.keys()):
            is_self = alliance == _PLAYER_SELF
            description += "Our units:\n" if is_self else "Enemy units:\n"

            # 获取该alliance的所有单位
            for original_tag in self.unit_manager.alliance_groups[alliance]:
                unit_info = self.unit_manager.unit_info[original_tag]
                _, unit_name = self.unit_manager.tag_registry[original_tag]

                # 构建状态信息
                health_info = f"Health: {unit_info.health:.1f}/{unit_info.max_health:.1f}"
                shield_info = f", Shield: {unit_info.shield:.1f}/{unit_info.max_shield:.1f}" if is_self else ""

                description += f"- {unit_name}: {health_info}{shield_info}\n"

            description += "\n"

        return description

    def get_original_tag(self, simplified_tag):
        """根据simplified_tag获取original_tag"""
        try:
            # 修正：使用正确的方法名
            original_tag = self.unit_manager.get_original_tag_by_simplified(simplified_tag)
            self.logger.debug(f"Looking up simplified_tag {simplified_tag} -> original_tag {original_tag}")

            if original_tag is None:
                self.logger.warning(f"No original tag found for simplified tag {simplified_tag}")
                return None

            # 验证单位是否有效
            if not self.unit_manager.is_valid_unit(original_tag):
                self.logger.warning(f"Unit with original tag {original_tag} is not valid/alive")
                return None

            return original_tag

        except Exception as e:
            self.logger.error(f"Error in get_original_tag: {e}")
            return None

    def _calculate_world_coordinates(self, grid_pos, unit_info, obs):
        """计算世界坐标的辅助函数"""
        try:
            if not hasattr(self, '_coordinate_cache'):
                self._coordinate_cache = {}

            cache_key = (tuple(grid_pos), unit_info.original_tag)
            if cache_key in self._coordinate_cache:
                return self._coordinate_cache[cache_key]

            # 将网格坐标转换为特征图坐标
            feature_x = grid_pos[0] * (self.feature_screen_size[0] / 10) + (self.feature_screen_size[0] / 20)
            feature_y = grid_pos[1] * (self.feature_screen_size[1] / 10) + (self.feature_screen_size[1] / 20)

            # 边界检查
            feature_x = max(0, min(feature_x, self.feature_screen_size[0] - 1))
            feature_y = max(0, min(feature_y, self.feature_screen_size[1] - 1))

            # 查找单位的feature位置
            for feature_unit in obs.observation.feature_units:
                if feature_unit.tag == unit_info.original_tag:
                    if feature_unit.x == 0 or feature_unit.y == 0:
                        return None, None

                    try:
                        # 计算缩放比例
                        scale_x = unit_info.x / feature_unit.x
                        scale_y = unit_info.y / feature_unit.y

                        # 计算目标world坐标
                        target_world_x = feature_x * scale_x
                        target_world_y = feature_y * scale_y

                        # 确保目标坐标在地图范围内
                        target_world_x = max(0, min(target_world_x, self.game_map_size[0] - 1))
                        target_world_y = max(0, min(target_world_y, self.game_map_size[1] - 1))

                        # 缓存结果
                        result = (target_world_x, target_world_y)
                        self._coordinate_cache[cache_key] = result
                        return result

                    except (ZeroDivisionError, ValueError) as e:
                        self._error_counts['coordinate_calculation'] += 1
                        if self._error_counts['coordinate_calculation'] >= self.max_errors:
                            self.logger.error(f"Repeated coordinate calculation errors: {e}")
                        return None, None

            return None, None

        except Exception as e:
            self.logger.error(f"Error calculating world coordinates: {e}")
            return None, None

    def create_move_action(self, simplified_tag, grid_pos, obs):
        """优化的移动动作创建函数"""
        try:
            # 类型检查
            if not isinstance(grid_pos, (list, tuple)) or len(grid_pos) != 2:
                self.logger.error(f"Invalid grid_pos format: {grid_pos}")
                return None

            if not isinstance(simplified_tag, int):
                self.logger.error(f"Invalid simplified_tag format: {simplified_tag}")
                return None

            # 检查缓存
            cache_key = (simplified_tag, tuple(grid_pos))
            if cache_key in self._move_cache:
                return self._move_cache[cache_key]

            # 基本验证
            original_tag = self.get_original_tag(simplified_tag)
            if original_tag is None:
                return None

            unit_info = self.unit_manager.unit_info.get(original_tag)
            if not unit_info or unit_info.health <= 0:
                return None

            # 计算世界坐标
            world_coords = self._calculate_world_coordinates(grid_pos, unit_info, obs)
            if world_coords is None or None in world_coords:
                return None

            # 创建移动动作
            action = actions.RAW_FUNCTIONS.Move_pt("now", original_tag, world_coords)

            # 缓存结果
            self._move_cache[cache_key] = action
            return action

        except Exception as e:
            self._error_counts['move_action'] += 1
            if self._error_counts['move_action'] >= self.max_errors:
                self.logger.error(f"Repeated move action errors: {e}")
            return None

    def create_smac_move_action(self, simplified_tag, direction, obs):
        """基于方向的移动"""
        original_tag = self.get_original_tag(simplified_tag)
        if original_tag is None:
            return None

        unit_info = self.unit_manager.unit_info.get(original_tag)
        if not unit_info or unit_info.health <= 0:  # 检查单位是否存活
            return None

        # 查找单位的feature位置
        for feature_unit in obs.observation.feature_units:
            if feature_unit.tag == original_tag:
                if feature_unit.x == 0 or feature_unit.y == 0:  # 避免除零错误
                    return None

                try:
                    # 计算feature map上的移动步长
                    step_size = self.feature_screen_size[0] / 10

                    # 根据方向计算新位置
                    new_feature_x = feature_unit.x
                    new_feature_y = feature_unit.y

                    if direction == UP:
                        new_feature_y -= step_size
                    elif direction == RIGHT:
                        new_feature_x += step_size
                    elif direction == DOWN:
                        new_feature_y += step_size
                    elif direction == LEFT:
                        new_feature_x -= step_size

                    # 边界检查
                    new_feature_x = max(0, min(new_feature_x, self.feature_screen_size[0] - 1))
                    new_feature_y = max(0, min(new_feature_y, self.feature_screen_size[1] - 1))

                    # 计算缩放比例
                    scale_x = unit_info.x / feature_unit.x
                    scale_y = unit_info.y / feature_unit.y

                    # 计算目标world坐标
                    target_world_x = new_feature_x * scale_x
                    target_world_y = new_feature_y * scale_y

                    # 确保目标坐标在地图范围内
                    target_world_x = max(0, min(target_world_x, self.game_map_size[0] - 1))
                    target_world_y = max(0, min(target_world_y, self.game_map_size[1] - 1))

                    return actions.RAW_FUNCTIONS.Move_pt("now", original_tag, (target_world_x, target_world_y))
                except (ZeroDivisionError, ValueError) as e:
                    self.logger.error(f"Error calculating move position: {e}")
                    return None

        return None

    def create_attack_action(self, attacker_simplified_tag, target_simplified_tag):
        """优化的攻击动作创建函数"""
        try:
            # 检查缓存
            cache_key = (attacker_simplified_tag, target_simplified_tag)
            if cache_key in self._attack_cache:
                return self._attack_cache[cache_key]

            # 获取原始tag
            attacker_original_tag = self.get_original_tag(attacker_simplified_tag)
            target_original_tag = self.get_original_tag(target_simplified_tag)

            if attacker_original_tag is None or target_original_tag is None:
                return None

            # 验证单位状态
            attacker = self.unit_manager.unit_info.get(attacker_original_tag)
            target = self.unit_manager.unit_info.get(target_original_tag)

            if not attacker or not target:
                return None

            if not attacker.alive or not target.alive:
                return None

            # 验证阵营
            if attacker.alliance != 1 or target.alliance != 4:
                return None

            # 创建攻击动作
            action = actions.RAW_FUNCTIONS.Attack_unit("now", attacker_original_tag, target_original_tag)

            # 缓存结果
            self._attack_cache[cache_key] = action
            return action

        except Exception as e:
            self._error_counts['attack_action'] += 1
            if self._error_counts['attack_action'] >= self.max_errors:
                self.logger.error(f"Repeated attack action errors: {e}")
            return None

    def get_new_position(self, current_pos, direction):
        """根据方向获取新位置"""
        x, y = current_pos
        if direction == UP:
            return (x, y - 1)
        elif direction == RIGHT:
            return (x + 1, y)
        elif direction == DOWN:
            return (x, y + 1)
        elif direction == LEFT:
            return (x - 1, y)
        return current_pos

    def validate_attack_command(self, attacker_tag, target_tag):
        """验证攻击命令"""
        # 获取原始tag
        attacker_original = self.unit_manager.get_original_tag_by_simplified(attacker_tag)
        target_original = self.unit_manager.get_original_tag_by_simplified(target_tag)

        if not attacker_original or not target_original:
            return False, "Invalid tag"

        # 验证单位存活和阵营
        attacker = self.unit_manager.unit_info.get(attacker_original)
        target = self.unit_manager.unit_info.get(target_original)

        if not attacker or not target:
            return False, "Unit not found"

        if attacker.alliance != 1 or target.alliance != 4:
            return False, "Invalid alliance"

        return True, ""