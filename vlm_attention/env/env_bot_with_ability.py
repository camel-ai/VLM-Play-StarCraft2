import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions, features
from collections import defaultdict
from vlm_attention.env.config import *

import logging
# 设置logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 玩家类型常量
_PLAYER_SELF = 1
_PLAYER_ENEMY = 4

# 方向常量
UP, RIGHT, DOWN, LEFT = range(4)


class UnitInfo:
    def __init__(self, unit, alliance, simplified_tag):
        print(f"\nInitializing UnitInfo:")
        print(f"Unit type: {unit.unit_type} ({get_unit_name(unit.unit_type)})")
        print(f"Alliance: {alliance}")
        print(f"Tag: {unit.tag} -> {simplified_tag}")

        # 基础属性
        self.original_tag = int(unit.tag)
        self.alliance = int(alliance)
        self.unit_type = int(unit.unit_type)
        self.simplified_tag = simplified_tag
        self.max_health = float(unit.health)
        self.max_shield = float(unit.shield)
        self.alive = True

        # 能力相关属性
        self.available_abilities = []  # 当前可用的技能
        self.ability_cooldowns = {}  # 技能冷却状态
        self.active_abilities = set()  # 正在使用的技能

        # 初始化状态
        self.update_status(unit)

    def update_status(self, unit, upgrades_completed=None):
        """更新单位状态和可用能力"""
        # 更新unit_type - 这是关键修改
        self.unit_type = int(unit.unit_type)

        # 更新基础状态
        self.health = float(unit.health)
        self.shield = float(unit.shield)
        self.energy = float(unit.energy)
        self.x = float(unit.x)
        self.y = float(unit.y)
        self.alive = self.health > 0

        # 更新技能状态
        self._update_abilities(unit, upgrades_completed)

    def _update_abilities(self, unit, upgrades_completed=None):
        """详细的技能更新逻辑"""
        print(f"\nUpdating abilities for unit: {get_unit_name(self.unit_type)}")

        # 清空当前技能列表
        self.available_abilities = []
        # 只为己方单位处理技能
        if self.alliance != _PLAYER_SELF:
            print(f"Skip abilities: enemy unit")
            return
        self.active_abilities.clear()

        # 1. 获取此单位类型的所有潜在技能
        potential_abilities = ABILITY_MANAGER.get_available_abilities(self.unit_type, upgrades_completed)
        print(f"Potential abilities based on unit type and upgrades: {potential_abilities}")

        # 2. 检查当前正在使用的技能
        if hasattr(unit, 'orders'):
            print(f"Unit has {len(unit.orders)} orders")
            for order in unit.orders:
                ability_id = getattr(order, 'ability_id', None)
                if ability_id:
                    print(f"Found active ability_id in orders: {ability_id}")
                    self.active_abilities.add(ability_id)

        if hasattr(unit, 'ability_id') and unit.ability_id:
            print(f"Unit has current ability_id: {unit.ability_id}")
            self.active_abilities.add(unit.ability_id)

        # 3. 为每个潜在技能创建详细信息
        for ability_name in potential_abilities:
            ability_info = ABILITY_MANAGER.get_ability_info(ability_name)
            if ability_info:
                # 检查技能是否在冷却中
                if ability_info['id'] in self.ability_cooldowns:
                    continue

                # 添加技能信息
                self.available_abilities.append(ability_info)
                print(f"Added available ability: {ability_name}")

    def get_ability_by_index(self, ability_idx):
        """通过索引获取技能信息"""
        for ability in self.available_abilities:
            if ability['ability_idx'] == ability_idx:
                return ability
        return None

    def has_ability(self, ability_name):
        """检查是否有特定技能"""
        return any(ability['name'] == ability_name for ability in self.available_abilities)

    def to_dict(self):
        """转换为字典格式以便序列化"""
        # 创建不包含函数对象的可序列化ability列表
        serializable_abilities = []
        for ability in self.available_abilities:
            # 只包含可序列化的字段
            serializable_ability = {
                'name': ability['name'],
                'id': ability['id'],
                'target_type': ability['target_type']
            }
            serializable_abilities.append(serializable_ability)

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
            'abilities': serializable_abilities,
            'active_abilities': list(self.active_abilities)
        }

    def __str__(self):
        """字符串表示"""
        abilities_str = ', '.join(a['name'] for a in self.available_abilities)
        return (f"{get_unit_name(self.unit_type)} [Tag: {self.simplified_tag}] "
                f"HP: {self.health}/{self.max_health} "
                f"Abilities: [{abilities_str}]")

class UnitManager:
    def __init__(self):
        self.unit_info = {}  # original_tag -> UnitInfo
        self.alliance_groups = defaultdict(list)  # alliance -> [original_tags]
        self.tag_registry = {}  # 新增：保存单位的永久性标识信息 {original_tag -> (simplified_tag, unit_name)}
        self.type_counters = {}  # 新增：记录每种单位类型的计数 {(unit_type, alliance) -> current_count}
        self.next_simplified_tag = 1  # 新增：用于生成新的simplified_tag
        self.initialized = False

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

    def update_units(self, units, upgrades_completed=None):
        """更新单位状态，保持标识符的一致性"""
        # 首次更新时初始化
        if not self.initialized:
            self.initialize_units(units)

        # 清除当前的alliance分组
        self.alliance_groups.clear()

        # 更新单位状态
        for unit in units:
            original_tag = int(unit.tag)
            alliance = int(unit.alliance)

            if alliance not in (_PLAYER_SELF, _PLAYER_ENEMY):
                continue

            # 获取已注册的单位信息
            unit_info = self.tag_registry.get(original_tag)
            if unit_info is None:
                # 如果是新出现的单位，为其分配新的标识
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

            # 更新或创建UnitInfo对象
            if original_tag not in self.unit_info:
                self.unit_info[original_tag] = UnitInfo(unit, alliance, unit_info[0])
            else:
                # 传入升级信息
                self.unit_info[original_tag].update_status(unit, upgrades_completed)

            # 更新alliance分组
            self.alliance_groups[alliance].append(original_tag)

        # 对alliance组内的单位按simplified_tag排序
        for alliance in self.alliance_groups:
            self.alliance_groups[alliance].sort(
                key=lambda x: self.tag_registry[x][0]
            )

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
        self.unit_manager = UnitManager()
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
        # 新增技能命令列表
        self.ability_commands = []  # [(caster_tag, ability_idx, target_info), ...]
        self.logger = logging.getLogger('Multimodal_bot')

    def step(self, obs):
        """处理每一步的观察并返回动作列表"""
        self.logger.info(f"\n=== Step {self.step_count} ===")

        # 第一步时记录单位的最大生命值和护盾
        if self.step_count == 1:
            for unit in obs.observation.raw_units:
                self.max_health_shield[unit.tag] = (unit.health, unit.shield)

        # 调用父类的step方法
        super(Multimodal_bot, self).step(obs)

        # 获取升级信息
        upgrades_completed = set()
        if hasattr(obs.observation, 'upgrades'):
            upgrades_array = obs.observation.upgrades
            if isinstance(upgrades_array, np.ndarray):
                upgrades_completed = set(upgrades_array.tolist())
            else:
                upgrades_completed = set(upgrades_array)

        # 更新单位管理器时传入升级信息
        self.unit_manager.update_units(obs.observation.raw_units, upgrades_completed)

        actions_list = []

        # 处理攻击命令
        for attacker_simplified_tag, target_simplified_tag in self.attack_commands:
            action = self.create_attack_action(attacker_simplified_tag, target_simplified_tag)
            if action is not None:
                actions_list.append(action)
                self.logger.info(f"Attack: Unit {attacker_simplified_tag} -> Unit {target_simplified_tag}")

        # 处理原始移动命令
        for unit_simplified_tag, grid_pos in self.original_move_commands:
            action = self.create_move_action(unit_simplified_tag, grid_pos, obs)
            if action is not None:
                actions_list.append(action)
                self.logger.info(f"Move: Unit {unit_simplified_tag} -> Position {grid_pos}")

        # 处理SMAC移动命令
        for unit_simplified_tag, direction in self.smac_move_commands:
            action = self.create_smac_move_action(unit_simplified_tag, direction, obs)
            if action is not None:
                actions_list.append(action)
                direction_name = ["UP", "RIGHT", "DOWN", "LEFT"][direction]
                self.logger.info(f"SMAC Move: Unit {unit_simplified_tag} -> {direction_name}")

        # 处理技能命令
        for caster_tag, ability_idx, target_info in self.ability_commands:
            action = self.create_ability_action(caster_tag, ability_idx, target_info, obs)
            if action is not None:
                actions_list.append(action)
                ability = self.unit_manager.unit_info[self.get_original_tag(caster_tag)].available_abilities[
                    ability_idx]
                self.logger.info(f"Ability: Unit {caster_tag} casts {ability['name']}")

        # 如果没有任何命令，执行空操作
        if not actions_list:
            actions_list.append(actions.RAW_FUNCTIONS.no_op())
            self.logger.info("No action - executing no_op")

        # 只记录非空操作的动作
        elif not (len(actions_list) == 1 and actions_list[0].function == actions.RAW_FUNCTIONS.no_op):
            self.logger.info(f"Step {self.step_count} actions: {actions_list}")

        # 清除本轮的命令
        self.attack_commands.clear()
        self.original_move_commands.clear()
        self.smac_move_commands.clear()
        self.ability_commands.clear()
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
        if hasattr(self, '_simplified_to_original_cache'):
            self._simplified_to_original_cache.clear()  # 清除缓存

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

    def add_ability_command(self, caster_simplified_tag, ability_idx, target_info):
        """添加技能命令

        Args:
            caster_simplified_tag (int): 施法者的简化ID
            ability_idx (int): 技能在available_abilities中的索引
            target_info (dict): 包含target_type和目标信息
        """
        # 检验施法者是否存在且属于己方
        caster_tag = self.get_original_tag(caster_simplified_tag)
        if caster_tag is None:
            return

        caster_info = self.unit_manager.unit_info[caster_tag]
        if caster_info.alliance != _PLAYER_SELF or caster_info.health <= 0:
            return

        # 检验技能是否可用
        if ability_idx >= len(caster_info.available_abilities):
            return

        # 获取技能信息
        ability = caster_info.available_abilities[ability_idx]

        # 验证目标类型是否匹配
        if ability['target_type'] != target_info['target_type']:
            return

        # 添加到命令列表(记得保持元组的三个元素)
        self.ability_commands.append((caster_simplified_tag, ability_idx, target_info))
    def get_raw_image_and_unit_info(self, obs):
        """获取原始图像和详细的单位信息"""
        print("\nProcessing raw units:")
        for unit in obs.observation.raw_units:
            if unit.alliance in (_PLAYER_SELF, _PLAYER_ENEMY):
                print(f"\nRaw unit info:")
                print(f"Tag: {unit.tag}")
                print(f"Type: {unit.unit_type} ({get_unit_name(unit.unit_type)})")
                print(f"Alliance: {unit.alliance}")
                print(f"Orders: {getattr(unit, 'orders', 'No orders')}")
                print(f"Current ability: {getattr(unit, 'ability_id', 'No ability')}")
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

        for alliance in sorted(self.unit_manager.alliance_groups.keys()):
            is_self = alliance == _PLAYER_SELF
            description += "Our units:\n" if is_self else "Enemy units:\n"

            for original_tag in self.unit_manager.alliance_groups[alliance]:
                unit_info = self.unit_manager.unit_info[original_tag]
                _, unit_name = self.unit_manager.tag_registry[original_tag]

                # 基础状态信息
                health_info = f"Health: {unit_info.health:.1f}/{unit_info.max_health:.1f}"
                shield_info = f", Shield: {unit_info.shield:.1f}/{unit_info.max_shield:.1f}" if is_self else ""

                # 技能信息
                ability_info = ""
                if is_self and unit_info.available_abilities:
                    ability_descriptions = []
                    for idx, ability in enumerate(unit_info.available_abilities):
                        target_type_name = next(name for name, value in TARGET_TYPE.items()
                                                if value == ability['target_type'])
                        ability_descriptions.append(
                            f"{ability['name']} (idx: {idx}, {target_type_name})"
                        )
                    ability_info = "\n  Available abilities: " + ", ".join(ability_descriptions)

                description += f"- {unit_name}: {health_info}{shield_info}{ability_info}\n"

            description += "\n"

        return description

    def get_original_tag(self, simplified_tag):
        """根据simplified_tag获取original_tag，使用缓存优化性能

        Args:
            simplified_tag: 简化的单位ID

        Returns:
            int or None: 原始tag，如果找不到则返回None
        """
        # 可以添加缓存字典作为类属性
        if not hasattr(self, '_simplified_to_original_cache'):
            self._simplified_to_original_cache = {}
            # 初始化缓存
            for original_tag, info in self.unit_manager.unit_info.items():
                self._simplified_to_original_cache[info.simplified_tag] = original_tag

        return self._simplified_to_original_cache.get(simplified_tag)

    def create_move_action(self, simplified_tag, grid_pos, obs):
        """基于网格的移动"""
        original_tag = self.get_original_tag(simplified_tag)
        if original_tag is None:
            return None

        unit_info = self.unit_manager.unit_info.get(original_tag)
        if not unit_info or unit_info.health <= 0:  # 检查单位是否存活
            return None

        # 将10x10网格转换为feature map坐标
        feature_x = grid_pos[0] * (self.feature_screen_size[0] / 10) + (self.feature_screen_size[0] / 20)
        feature_y = grid_pos[1] * (self.feature_screen_size[1] / 10) + (self.feature_screen_size[1] / 20)

        # 边界检查
        feature_x = max(0, min(feature_x, self.feature_screen_size[0] - 1))
        feature_y = max(0, min(feature_y, self.feature_screen_size[1] - 1))

        # 查找单位的feature位置
        for feature_unit in obs.observation.feature_units:
            if feature_unit.tag == original_tag:
                if feature_unit.x == 0 or feature_unit.y == 0:  # 避免除零错误
                    return None

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

                    return actions.RAW_FUNCTIONS.Move_pt("now", original_tag, (target_world_x, target_world_y))
                except (ZeroDivisionError, ValueError) as e:
                    self.logger.error(f"Error calculating move position: {e}")
                    return None

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
        """创建攻击动作
        Args:
            attacker_simplified_tag: 攻击者的简化ID
            target_simplified_tag: 目标的简化ID
        """
        attacker_original_tag = self.get_original_tag(attacker_simplified_tag)
        target_original_tag = self.get_original_tag(target_simplified_tag)

        if attacker_original_tag is None or target_original_tag is None:
            return None

        target_unit = self.unit_manager.unit_info[target_original_tag]
        return actions.RAW_FUNCTIONS.Attack_unit("now", attacker_original_tag, target_original_tag)

    def create_ability_action(self, caster_simplified_tag, ability_idx, target_info, obs):
        """创建技能动作

        Args:
            caster_simplified_tag: 施法者的简化ID
            ability_idx: ability的索引
            target_info: 包含target_type和具体目标信息的字典
            obs: 当前的观察
        """
        # 获取施法者原始tag
        caster_tag = self.get_original_tag(caster_simplified_tag)
        if caster_tag is None:
            return None

        caster_info = self.unit_manager.unit_info.get(caster_tag)
        if not caster_info:
            return None

        # 获取ability名称和信息
        for ability in caster_info.available_abilities:
            if ability['target_type'] == target_info['target_type']:
                try:
                    # 使用ABILITY_MANAGER创建技能动作
                    if target_info['target_type'] == TARGET_TYPE["QUICK"]:
                        return ABILITY_MANAGER.create_raw_ability_action(
                            ability['name'],
                            caster_tag
                        )

                    elif target_info['target_type'] == TARGET_TYPE["POINT"]:
                        world_pos = self._convert_grid_to_world_pos(
                            target_info['position'],
                            caster_info,
                            obs
                        )
                        if world_pos[0] is not None:
                            return ABILITY_MANAGER.create_raw_ability_action(
                                ability['name'],
                                caster_tag,
                                target_point=world_pos
                            )

                    elif target_info['target_type'] == TARGET_TYPE["UNIT"]:
                        target_tag = self.get_original_tag(target_info['target_unit'])
                        if target_tag is not None:
                            return ABILITY_MANAGER.create_raw_ability_action(
                                ability['name'],
                                caster_tag,
                                target_unit_tag=target_tag
                            )

                    elif target_info['target_type'] == TARGET_TYPE["AUTO"]:
                        return ABILITY_MANAGER.create_raw_ability_action(
                            ability['name'],
                            caster_tag,
                            target_type=TARGET_TYPE["AUTO"]
                        )

                except Exception as e:
                    self.logger.error(f"Error creating ability action: {e}")
                    return None

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
    def _convert_grid_to_world_pos(self, grid_pos, caster_info, obs):
        """将网格坐标转换为游戏世界坐标"""
        # 将10x10网格转换为feature map坐标
        feature_x = grid_pos[0] * (self.feature_screen_size[0] / 10) + (self.feature_screen_size[0] / 20)
        feature_y = grid_pos[1] * (self.feature_screen_size[1] / 10) + (self.feature_screen_size[1] / 20)

        # 边界检查
        feature_x = max(0, min(feature_x, self.feature_screen_size[0] - 1))
        feature_y = max(0, min(feature_y, self.feature_screen_size[1] - 1))

        # 查找施法者的feature位置用于计算缩放比例
        for feature_unit in obs.observation.feature_units:
            if feature_unit.tag == caster_info.original_tag:
                if feature_unit.x == 0 or feature_unit.y == 0:  # 避免除零错误
                    return None, None

                try:
                    # 计算缩放比例
                    scale_x = caster_info.x / feature_unit.x
                    scale_y = caster_info.y / feature_unit.y

                    # 计算目标world坐标
                    target_world_x = feature_x * scale_x
                    target_world_y = feature_y * scale_y

                    # 确保目标坐标在地图范围内
                    target_world_x = max(0, min(target_world_x, self.game_map_size[0] - 1))
                    target_world_y = max(0, min(target_world_y, self.game_map_size[1] - 1))

                    return target_world_x, target_world_y
                except (ZeroDivisionError, ValueError):
                    return None, None

        return None, None
