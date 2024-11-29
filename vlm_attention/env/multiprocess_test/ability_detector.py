from pysc2.lib import units
from pysc2.lib import upgrades
from typing import List, Dict, Set
import numpy as np


class AbilityDetector:
    def __init__(self):
        # 定义微操相关的技能ID映射
        self.MICRO_ABILITIES = {
            # Terran Bio Abilities
            'STIM_MARINE': 380,  # Effect_Stim_Marine_quick
            'STIM_MARAUDER': 253,  # Effect_Stim_Marauder_quick
            'SNIPE': 2714,  # Effect_GhostSnipe_unit
            'EMP': 1628,  # Effect_EMP_pt
            'CLOAK': 3676,  # Behavior_CloakOn_quick
            'DECLOAK': 3677,  # Behavior_CloakOff_quick

            # Terran Mech Abilities
            'SIEGE_MODE': 388,  # Morph_SiegeMode_quick
            'UNSIEGE': 390,  # Morph_Unsiege_quick
            'ASSAULT_MODE': 403,  # Morph_VikingAssaultMode_quick
            'FIGHTER_MODE': 405,  # Morph_VikingFighterMode_quick
            'HIGH_IMPACT_MODE': 2362,  # Morph_ThorHighImpactMode_quick
            'EXPLOSIVE_MODE': 2364,  # Morph_ThorExplosiveMode_quick
            'MINE_ATTACK': 2099,  # Effect_WidowMineAttack_pt

            # Terran Support Abilities
            'HEAL': 386,  # Effect_Heal_unit/autocast
            'BOOST': 2116,  # Effect_MedivacIgniteAfterburners_quick
            'AUTO_TURRET': 1764,  # Effect_AutoTurret_pt
            'YAMATO': 401,  # Effect_YamatoGun_unit

            # Transport Abilities
            'LOAD': 3668,  # Load_unit
            'LOAD_BUNKER': 407,  # Load_Bunker_unit
            'LOAD_MEDIVAC': 394,  # Load_Medivac_unit
            'UNLOAD_ALL': 3664,  # UnloadAll_quick
            'UNLOAD_AT': 3669,  # UnloadAllAt_pt
            'UNLOAD_UNIT': 3796,  # UnloadUnit_quick

            # Protoss Gateway Abilities
            'BLINK': 3687,  # Effect_Blink_pt
            'GUARDIAN_SHIELD': 76,  # Effect_GuardianShield_quick
            'FORCE_FIELD': 1526,  # Effect_ForceField_pt
            'PSI_STORM': 1036,  # Effect_PsiStorm_pt
            'FEEDBACK': 140,  # Effect_Feedback_unit

            # Protoss Robotic Abilities
            'IMMORTAL_BARRIER': 2328,  # Effect_ImmortalBarrier_quick/autocast
            'PRISM_PHASE': 1528,  # Morph_WarpPrismPhasingMode_quick
            'PRISM_TRANSPORT': 1530,  # Morph_WarpPrismTransportMode_quick

            # Protoss Stargate Abilities
            'GRAVITON_BEAM': 173,  # Effect_GravitonBeam_unit
            'REVELATION': 2146,  # Effect_OracleRevelation_pt
            'PRISMATIC_ALIGNMENT': 2393,  # Effect_VoidRayPrismaticAlignment_quick

            # Zerg Ground Abilities
            'EXPLODE': 42,  # Effect_Explode_quick
            'FUNGAL': 74,  # Effect_FungalGrowth_pt
            'NEURAL_PARASITE': 249,  # Effect_NeuralParasite_unit
            'BURROW': 3661,  # BurrowDown_quick
            'UNBURROW': 3662,  # BurrowUp_quick

            # Zerg Air Abilities
            'ABDUCT': 2067,  # Effect_Abduct_unit
            'BLIND': 2063,  # Effect_BlindingCloud_pt
            'CONSUME': 2073,  # Effect_ViperConsume_unit
        }

        # 更新单位类型与技能的对应关系
        self.UNIT_ABILITY_MAP = {
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
                'base': {'SNIPE', 'EMP'},
                'upgraded': {'CLOAK', 'DECLOAK'}
            },

            # Terran Mech Units
            units.Terran.Thor: {
                'base': {'HIGH_IMPACT_MODE', 'EXPLOSIVE_MODE'},
                'upgraded': set()
            },
            units.Terran.ThorHighImpactMode: {
                'base': {'HIGH_IMPACT_MODE', 'EXPLOSIVE_MODE'},
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
                'base': {'FIGHTER_MODE'},
                'upgraded': set()
            },
            units.Terran.VikingFighter: {
                'base': {'ASSAULT_MODE'},
                'upgraded': set()
            },
            units.Terran.WidowMine: {
                'base': {'MINE_ATTACK', 'BURROW'},
                'upgraded': set()
            },

            # Terran Support Units
            units.Terran.Medivac: {
                'base': {
                    'HEAL',
                    'BOOST',
                    'LOAD',
                    'LOAD_MEDIVAC',
                    'UNLOAD_ALL',
                    'UNLOAD_AT',
                    'UNLOAD_UNIT'
                },
                'upgraded': set()
            },
            units.Terran.Bunker: {
                'base': {
                    'LOAD',
                    'LOAD_BUNKER',
                    'UNLOAD_ALL',
                    'UNLOAD_AT',
                    'UNLOAD_UNIT'
                },
                'upgraded': set()
            },
            units.Terran.Battlecruiser: {
                'base': {'YAMATO'},
                'upgraded': set()
            },

            # Protoss Gateway Units
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

            # Protoss Robotic Units
            units.Protoss.Immortal: {
                'base': {'IMMORTAL_BARRIER'},
                'upgraded': set()
            },
            units.Protoss.WarpPrism: {
                'base': {
                    'PRISM_PHASE',
                    'LOAD',
                    'UNLOAD_ALL',
                    'UNLOAD_AT',
                    'UNLOAD_UNIT'
                },
                'upgraded': set()
            },
            units.Protoss.WarpPrismPhasing: {
                'base': {'PRISM_TRANSPORT'},
                'upgraded': set()
            },

            # Protoss Stargate Units
            units.Protoss.Phoenix: {
                'base': {'GRAVITON_BEAM'},
                'upgraded': set()
            },
            units.Protoss.Oracle: {
                'base': {'REVELATION'},
                'upgraded': set()
            },
            units.Protoss.VoidRay: {
                'base': {'PRISMATIC_ALIGNMENT'},
                'upgraded': set()
            },

            # Zerg Ground Units
            units.Zerg.Baneling: {
                'base': {'EXPLODE', 'BURROW'},
                'upgraded': set()
            },
            units.Zerg.Infestor: {
                'base': {'FUNGAL', 'NEURAL_PARASITE', 'BURROW'},
                'upgraded': set()
            },
            units.Zerg.Queen: {
                'base': {'BURROW'},
                'upgraded': set()
            },
            units.Zerg.Roach: {
                'base': {'BURROW'},
                'upgraded': set()
            },
            units.Zerg.Zergling: {
                'base': {'BURROW'},
                'upgraded': set()
            },

            # Zerg Air Units
            units.Zerg.Viper: {
                'base': {'ABDUCT', 'BLIND', 'CONSUME'},
                'upgraded': set()
            },
            units.Zerg.Overlord: {
                'base': {
                    'LOAD',
                    'UNLOAD_ALL',
                    'UNLOAD_AT',
                    'UNLOAD_UNIT'
                },
                'upgraded': set()
            }
        }

        # 更新升级与技能的对应关系
        self.UPGRADE_ABILITY_MAP = {
            15: {'STIM_MARINE', 'STIM_MARAUDER'},  # Stimpack
            25: {'CLOAK'},  # Personal Cloaking
            87: {'BLINK'},  # Blink
            52: {'PSI_STORM'},  # Psi Storm
            101: {'NEURAL_PARASITE'},  # Neural Parasite
            64: {'BURROW', 'UNBURROW'},  # Burrow
        }

    def get_available_abilities(self, obs) -> Dict[str, Set[str]]:
        """
        基于当前游戏状态检测可用的微操技能

        Args:
            obs: pysc2的observation对象

        Returns:
            Dict[str, Set[str]]: 按单位类型分类的可用技能集合
        """
        available_abilities = {}

        # 获取升级状态
        player_upgrades = set()
        if hasattr(obs.observation, 'upgrades'):
            upgrades_array = obs.observation.upgrades
            if isinstance(upgrades_array, np.ndarray):
                player_upgrades = set(upgrades_array.tolist())
            else:
                player_upgrades = set(upgrades_array)

        # 检查每个友方单位
        if obs.observation.raw_units is not None:
            for unit in obs.observation.raw_units:
                if unit.alliance == 1:  # 1表示友方单位
                    try:
                        unit_enum = units.get_unit_type(unit.unit_type)
                        if unit_enum in self.UNIT_ABILITY_MAP:
                            unit_name = unit_enum.name
                            if unit_name not in available_abilities:
                                available_abilities[unit_name] = set()

                            # 添加基础技能
                            available_abilities[unit_name].update(
                                self.UNIT_ABILITY_MAP[unit_enum]['base']
                            )

                            # 只有在有对应升级的情况下才添加需要升级的技能
                            for upgrade_id, abilities in self.UPGRADE_ABILITY_MAP.items():
                                if upgrade_id in player_upgrades:
                                    upgraded_abilities = self.UNIT_ABILITY_MAP[unit_enum]['upgraded']
                                    available_abilities[unit_name].update(
                                        abilities.intersection(upgraded_abilities)
                                    )
                    except ValueError:
                        continue  # 忽略未知单位类型

        return available_abilities

    def print_available_abilities(self, obs) -> None:
        """
        打印当前可用的所有微操技能
        """
        abilities = self.get_available_abilities(obs)
        print("\n=== Available Micro Abilities ===")
        for unit_type, unit_abilities in sorted(abilities.items()):
            if unit_abilities:  # 只打印有可用技能的单位
                print(f"\nUnit Type: {unit_type}")
                for ability in sorted(unit_abilities):
                    ability_id = self.MICRO_ABILITIES.get(ability, "Unknown")
                    print(f"- {ability} (ID: {ability_id})")

    def print_debug_info(self, obs) -> None:
        """
        打印调试信息
        """
        print("\n=== Debug Information ===")

        # 打印升级信息
        print("Current Upgrades:")
        if hasattr(obs.observation, 'upgrades'):
            upgrades_array = obs.observation.upgrades
            if isinstance(upgrades_array, np.ndarray) and upgrades_array.size > 0:
                for upgrade_id in upgrades_array:
                    try:
                        upgrade_enum = upgrades.Upgrades(upgrade_id)
                        print(f"- Upgrade ID: {upgrade_id} ({upgrade_enum.name})")
                    except ValueError:
                        print(f"- Unknown Upgrade ID: {upgrade_id}")
            else:
                print("No upgrades found")
        else:
            print("No upgrade information available")

        # 打印单位信息
        print("\nCurrent Units:")
        if obs.observation.raw_units is not None:
            for unit in obs.observation.raw_units:
                if unit.alliance == 1:  # 友方单位
                    try:
                        unit_enum = units.get_unit_type(unit.unit_type)
                        print(f"- Type: {unit.unit_type} ({unit_enum.name}), Alliance: {unit.alliance}")
                    except ValueError:
                        print(f"- Unknown Type: {unit.unit_type}, Alliance: {unit.alliance}")
        else:
            print("No unit information available")