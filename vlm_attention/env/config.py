from pysc2.lib.units import Neutral, Protoss, Terran, Zerg
from pysc2.lib.actions import RAW_FUNCTIONS
from pysc2.lib import actions, units, upgrades
import logging

"""
Config of the entire environment, including:
action types, direction constants, 
player types, colors, 
target types, logger, 
ability manager, format unit name, 
get unit name, ability manager.

"""
# Action types
_NO_OP = 0
_MOVE = 1
_ATTACK = 2
_ABILITY = 3

# Direction constants
UP, RIGHT, DOWN, LEFT = range(4)

# Player types
_PLAYER_SELF = 1
_PLAYER_ENEMY = 4

# Colors (BGR format)
COLORS = {
    "self_color": [0, 255, 0],  # Green
    "enemy_color": [255, 255, 0]  # Yellow
}

# Target types for abilities
TARGET_TYPE = {
    "QUICK": 0,  # No target needed (e.g. Stim)
    "POINT": 1,  # Target location (e.g. Force Field)
    "UNIT": 2,  # Target unit (e.g. Snipe)
    "AUTO": 3  # Autocast (e.g. Heal)
}

logger = logging.getLogger(__name__)

class AbilityManager:
    """用于管理游戏中的技能系统"""

    def __init__(self):
        # 将技能映射到对应的RAW_FUNCTIONS
        self.ability_function_map = {
            # Terran Bio Abilities
            'STIM_MARINE': (RAW_FUNCTIONS.Effect_Stim_Marine_quick, TARGET_TYPE["QUICK"]),
            'STIM_MARAUDER': (RAW_FUNCTIONS.Effect_Stim_Marauder_quick, TARGET_TYPE["QUICK"]),
            'GHOST_SNIPE': (RAW_FUNCTIONS.Effect_GhostSnipe_unit, TARGET_TYPE["UNIT"]),
            'GHOST_EMP': (RAW_FUNCTIONS.Effect_EMP_pt, TARGET_TYPE["POINT"]),
            'GHOST_CLOAK': (RAW_FUNCTIONS.Behavior_CloakOn_Ghost_quick, TARGET_TYPE["QUICK"]),
            'GHOST_DECLOAK': (RAW_FUNCTIONS.Behavior_CloakOff_Ghost_quick, TARGET_TYPE["QUICK"]),

            # Terran Mech Abilities
            'SIEGE_MODE': (RAW_FUNCTIONS.Morph_SiegeMode_quick, TARGET_TYPE["QUICK"]),
            'UNSIEGE': (RAW_FUNCTIONS.Morph_Unsiege_quick, TARGET_TYPE["QUICK"]),
            'VIKING_ASSAULT': (RAW_FUNCTIONS.Morph_VikingAssaultMode_quick, TARGET_TYPE["QUICK"]),
            'VIKING_FIGHTER': (RAW_FUNCTIONS.Morph_VikingFighterMode_quick, TARGET_TYPE["QUICK"]),
            'THOR_HIGH_IMPACT': (RAW_FUNCTIONS.Morph_ThorHighImpactMode_quick, TARGET_TYPE["QUICK"]),
            'THOR_EXPLOSIVE': (RAW_FUNCTIONS.Morph_ThorExplosiveMode_quick, TARGET_TYPE["QUICK"]),

            # Terran Support Abilities
            'MEDIVAC_HEAL': (RAW_FUNCTIONS.Effect_Heal_unit, TARGET_TYPE["UNIT"]),
            'MEDIVAC_BOOST': (RAW_FUNCTIONS.Effect_MedivacIgniteAfterburners_quick, TARGET_TYPE["QUICK"]),
            'LOAD': (RAW_FUNCTIONS.Load_unit, TARGET_TYPE["UNIT"]),
            'LOAD_BUNKER': (RAW_FUNCTIONS.Load_Bunker_unit, TARGET_TYPE["UNIT"]),
            'LOAD_MEDIVAC': (RAW_FUNCTIONS.Load_Medivac_unit, TARGET_TYPE["UNIT"]),
            'UNLOAD_ALL': (RAW_FUNCTIONS.UnloadAll_quick, TARGET_TYPE["QUICK"]),
            'UNLOAD_AT': (RAW_FUNCTIONS.UnloadAllAt_pt, TARGET_TYPE["POINT"]),
            'UNLOAD_UNIT': (RAW_FUNCTIONS.UnloadUnit_quick, TARGET_TYPE["QUICK"]),

            # Protoss Gateway Abilities
            'BLINK': (RAW_FUNCTIONS.Effect_Blink_pt, TARGET_TYPE["POINT"]),
            'GUARDIAN_SHIELD': (RAW_FUNCTIONS.Effect_GuardianShield_quick, TARGET_TYPE["QUICK"]),
            'FORCE_FIELD': (RAW_FUNCTIONS.Effect_ForceField_pt, TARGET_TYPE["POINT"]),
            'PSI_STORM': (RAW_FUNCTIONS.Effect_PsiStorm_pt, TARGET_TYPE["POINT"]),
            'FEEDBACK': (RAW_FUNCTIONS.Effect_Feedback_unit, TARGET_TYPE["UNIT"]),

            # Zerg Abilities
            'BANELING_EXPLODE': (RAW_FUNCTIONS.Effect_Explode_quick, TARGET_TYPE["QUICK"]),
            'BURROW': (RAW_FUNCTIONS.BurrowDown_quick, TARGET_TYPE["QUICK"]),
            'UNBURROW': (RAW_FUNCTIONS.BurrowUp_quick, TARGET_TYPE["QUICK"])
        }

        # 单位技能映射表
        self.unit_ability_map = {
            # Terran Bio Units
            units.Terran.Marine: {
                'base': set(),
                'upgraded': {'STIM_MARINE'}
            },
            units.Terran.Marauder: {
                'base': set(),
                'upgraded': {'STIM_MARAUDER'}
            },
            units.Terran.Ghost: {
                'base': {'GHOST_SNIPE', 'GHOST_EMP'},
                'upgraded': {'GHOST_CLOAK', 'GHOST_DECLOAK'}
            },

            # Terran Mech Units
            units.Terran.Thor: {
                'base': {'THOR_HIGH_IMPACT'},
                'upgraded': set()
            },
            units.Terran.ThorHighImpactMode: {
                'base': {'THOR_EXPLOSIVE'},
                'upgraded': set()
            },
            units.Terran.SiegeTank: {
                'base': {'SIEGE_MODE'},
                'upgraded': set()
            },
            units.Terran.SiegeTankSieged: {
                'base': {'UNSIEGE'},
                'upgraded': set()
            },
            units.Terran.VikingAssault: {
                'base': {'VIKING_FIGHTER'},
                'upgraded': set()
            },
            units.Terran.VikingFighter: {
                'base': {'VIKING_ASSAULT'},
                'upgraded': set()
            },

            # Terran Support Units
            units.Terran.Medivac: {
                'base': {
                    'MEDIVAC_HEAL',
                    'MEDIVAC_BOOST',
                    'LOAD',
                    'LOAD_MEDIVAC',
                    'UNLOAD_ALL',
                    'UNLOAD_AT',
                    'UNLOAD_UNIT'
                },
                'upgraded': set()
            },

            # Protoss Units
            units.Protoss.Stalker: {
                'base': set(),
                'upgraded': {'BLINK'}
            },
            units.Protoss.Sentry: {
                'base': {'GUARDIAN_SHIELD', 'FORCE_FIELD'},
                'upgraded': set()
            },
            units.Protoss.HighTemplar: {
                'base': {'FEEDBACK'},
                'upgraded': {'PSI_STORM'}
            },

            # Zerg Units
            units.Zerg.Baneling: {
                'base': {'BANELING_EXPLODE', 'BURROW'},
                'upgraded': set()
            }
        }

        # 升级能力映射表
        self.upgrade_ability_map = {
            # Terran Upgrades
            upgrades.Upgrades.CombatShield: {'STIM_MARINE', 'STIM_MARAUDER'},
            upgrades.Upgrades.PersonalCloaking: {'GHOST_CLOAK', 'GHOST_DECLOAK'},

            # Protoss Upgrades
            upgrades.Upgrades.Blink: {'BLINK'},
            upgrades.Upgrades.PsiStorm: {'PSI_STORM'},

            # Zerg Upgrades
            upgrades.Upgrades.Burrow: {'BURROW', 'UNBURROW'}
        }
        # 添加ability ID映射
        self.ability_ids = {
            # Terran Bio Abilities
            'STIM_MARINE': 380,
            'STIM_MARAUDER': 253,
            'GHOST_SNIPE': 2714,
            'GHOST_EMP': 1628,
            'GHOST_CLOAK': 3676,
            'GHOST_DECLOAK': 3677,

            # Terran Mech Abilities
            'SIEGE_MODE': 388,
            'UNSIEGE': 390,
            'VIKING_ASSAULT': 403,
            'VIKING_FIGHTER': 405,
            'THOR_HIGH_IMPACT': 2362,
            'THOR_EXPLOSIVE': 2364,

            # Terran Support Abilities
            'MEDIVAC_HEAL': 386,
            'MEDIVAC_BOOST': 2116,
            'LOAD': 3668,
            'LOAD_BUNKER': 407,
            'LOAD_MEDIVAC': 394,
            'UNLOAD_ALL': 3664,
            'UNLOAD_AT': 3669,
            'UNLOAD_UNIT': 3796,

            # Protoss Gateway Abilities
            'BLINK': 3687,
            'GUARDIAN_SHIELD': 76,
            'FORCE_FIELD': 1526,
            'PSI_STORM': 1036,
            'FEEDBACK': 140,

            # Zerg Abilities
            'BANELING_EXPLODE': 42,
            'BURROW': 3661,
            'UNBURROW': 3662
        }

        # 反向映射
        self.id_to_name = {v: k for k, v in self.ability_ids.items()}

    def get_ability_id(self, ability_name: str) -> int:
        """获取技能ID

        Args:
            ability_name: 技能名称

        Returns:
            int: 技能ID，如果不存在则返回None
        """
        return self.ability_ids.get(ability_name)

    def get_ability_name(self, ability_id: int) -> str:
        """通过ID获取技能名称

        Args:
            ability_id: 技能ID

        Returns:
            str: 技能名称，如果不存在则返回None
        """
        return self.id_to_name.get(ability_id)

    def get_ability_info(self, ability_name: str) -> dict:
        """获取技能完整信息

        Args:
            ability_name: 技能名称

        Returns:
            dict: 包含技能ID、名称、目标类型等信息的字典，不包含函数对象
        """
        ability_id = self.get_ability_id(ability_name)
        if ability_id is None:
            return None

        raw_func, target_type = self.get_function_and_type(ability_name)
        if raw_func is None:
            return None

        # 不再包含raw_function
        return {
            'name': ability_name,
            'id': ability_id,
            'target_type': target_type
        }

    def get_available_abilities(self, unit_type, upgrades_completed=None):
        """获取单位当前可用的能力"""
        if upgrades_completed is None:
            upgrades_completed = set()

        try:
            unit_enum = units.get_unit_type(unit_type)
            if unit_enum not in self.unit_ability_map:
                return set()

            # 获取基础能力
            abilities = self.unit_ability_map[unit_enum]['base'].copy()

            # 添加已解锁的升级能力
            upgraded_abilities = self.unit_ability_map[unit_enum]['upgraded']
            for upgrade_id in upgrades_completed:
                try:
                    upgrade_enum = upgrades.Upgrades(upgrade_id)
                    if upgrade_enum in self.upgrade_ability_map:
                        abilities.update(
                            upgraded_abilities.intersection(
                                self.upgrade_ability_map[upgrade_enum])
                        )
                except ValueError:
                    continue

            return abilities
        except ValueError:
            return set()

    def get_function_and_type(self, ability_name):
        """获取能力对应的RAW_FUNCTIONS函数和目标类型"""
        return self.ability_function_map.get(ability_name, (None, None))

    def create_raw_ability_action(self, ability_name, unit_tags,
                                  target_type=None, target_point=None,
                                  target_unit_tag=None, queued=0):
        """创建原始技能动作

        Args:
            ability_name: 技能名称
            unit_tags: 施放技能的单位tag(list或单个tag)
            target_type: 目标类型(可选,如不提供则使用默认)
            target_point: 目标位置(可选)
            target_unit_tag: 目标单位tag(可选)
            queued: 是否排队(默认0)

        Returns:
            action or None: 返回创建的动作或None(如果创建失败)
        """
        raw_func, default_target_type = self.get_function_and_type(ability_name)
        if raw_func is None:
            return None

        # 确保unit_tags是列表
        if not isinstance(unit_tags, (list, tuple)):
            unit_tags = [unit_tags]

        target_type = target_type if target_type is not None else default_target_type
        try:
            if target_type == TARGET_TYPE["QUICK"]:
                return raw_func("now", unit_tags)

            elif target_type == TARGET_TYPE["POINT"] and target_point is not None:
                return raw_func("now", unit_tags, target_point)

            elif target_type == TARGET_TYPE["UNIT"] and target_unit_tag is not None:
                return raw_func("now", unit_tags, target_unit_tag)

            elif target_type == TARGET_TYPE["AUTO"]:
                # 对于autocast类型，直接使用raw_func
                return raw_func("now", unit_tags)

        except Exception as e:
            print(f"Error creating ability action for {ability_name}: {e}")

        return None

    def debug_unit_abilities(self, unit_type):
        """打印单位的所有可用技能"""
        try:
            unit_enum = units.get_unit_type(unit_type)
            base_abilities = self.unit_ability_map.get(unit_enum, {}).get('base', set())
            upgraded_abilities = self.unit_ability_map.get(unit_enum, {}).get('upgraded', set())
            
            logger.info(f"Unit type {unit_type} ({unit_enum.name}) abilities:")
            logger.info(f"Base abilities: {base_abilities}")
            logger.info(f"Upgraded abilities: {upgraded_abilities}")
            
            return base_abilities, upgraded_abilities
        except ValueError:
            logger.error(f"Invalid unit type: {unit_type}")
            return set(), set()


def format_unit_name(enum_name: str) -> str:
    """格式化单位名称"""
    formatted = ''.join(' ' + c if c.isupper() and i > 0 else c
                        for i, c in enumerate(enum_name))
    return formatted.strip()


def get_unit_name(unit_type: int) -> str:
    """获取单位名称"""
    for race in [Protoss, Terran, Zerg, Neutral]:
        try:
            unit_enum = race(unit_type)
            return format_unit_name(unit_enum.name)
        except ValueError:
            continue
    return f"Unknown Unit {unit_type}"


# 创建全局AbilityManager实例
ABILITY_MANAGER = AbilityManager()
