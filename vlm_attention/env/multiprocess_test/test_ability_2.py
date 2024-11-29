import os
import multiprocessing
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features,units
from pysc2.lib.actions import RAW_FUNCTIONS
from absl import app, flags
import logging
from vlm_attention.env.multiprocess_test.ability_detector import AbilityDetector
from collections import defaultdict
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS


class UnitTracker:
    def __init__(self, tag, unit_type, alliance, x=0, y=0):
        self.tag = tag
        self.unit_type = unit_type
        self.alliance = alliance
        self.x = x
        self.y = y
        self.cargo_space_taken = 0  # 用于记录运输机已使用的空间


class AbilityTestAgent(base_agent.BaseAgent):
    def __init__(self):
        super(AbilityTestAgent, self).__init__()
        self.abilities_checked = False
        self.ability_detector = AbilityDetector()
        self.step_count = 0
        self.units = {}  # tag -> UnitTracker

        # 单位类型常量
        self.UNIT_TYPES = {
            'MARINE': units.Terran.Marine,
            'MARAUDER': units.Terran.Marauder,
            'MEDIVAC': units.Terran.Medivac,
            'VIKING_ASSAULT': units.Terran.VikingAssault,
            'VIKING_FIGHTER': units.Terran.VikingFighter,
            'SIEGE_TANK': units.Terran.SiegeTank,
            'SIEGE_TANK_SIEGED': units.Terran.SiegeTankSieged,
            'THOR': units.Terran.Thor,
            'THOR_HIGH_IMPACT': units.Terran.ThorHighImpactMode
        }

    def update_unit_tracking(self, obs):
        """更新单位追踪状态"""
        current_tags = set()

        for unit in obs.observation.raw_units:
            if unit.alliance == 1:  # 只追踪我方单位
                current_tags.add(unit.tag)
                if unit.tag not in self.units:
                    self.units[unit.tag] = UnitTracker(
                        tag=unit.tag,
                        unit_type=unit.unit_type,
                        alliance=unit.alliance,
                        x=unit.x,
                        y=unit.y
                    )

                # 更新单位状态
                unit_tracker = self.units[unit.tag]
                unit_tracker.unit_type = unit.unit_type
                unit_tracker.x = unit.x
                unit_tracker.y = unit.y
                if hasattr(unit, 'cargo_space_taken'):
                    unit_tracker.cargo_space_taken = unit.cargo_space_taken

        # 移除不再存在的单位
        for tag in list(self.units.keys()):
            if tag not in current_tags:
                del self.units[tag]

    def get_units_by_type(self, unit_type):
        """根据单位类型获取单位"""
        return [unit for unit in self.units.values() if unit.unit_type == unit_type]

    def execute_actions(self, obs):
        """基于当前游戏状态执行动作"""
        actions_to_execute = []

        # 获取当前可用的能力
        available_abilities = self.ability_detector.get_available_abilities(obs)

        # 处理生物单位技能
        if "Marine" in available_abilities and "STIM_MARINE" in available_abilities["Marine"]:
            for unit in self.get_units_by_type(self.UNIT_TYPES['MARINE'].value):
                actions_to_execute.append(RAW_FUNCTIONS.Effect_Stim_Marine_quick("now", unit.tag))
                logger.info(f"Marine {unit.tag} using Stim Pack")

        if "Marauder" in available_abilities and "STIM_MARAUDER" in available_abilities["Marauder"]:
            for unit in self.get_units_by_type(self.UNIT_TYPES['MARAUDER'].value):
                actions_to_execute.append(RAW_FUNCTIONS.Effect_Stim_Marauder_quick("now", unit.tag))
                logger.info(f"Marauder {unit.tag} using Stim Pack")

        # 处理维京战机变形
        vikings_assault = self.get_units_by_type(self.UNIT_TYPES['VIKING_ASSAULT'].value)
        vikings_fighter = self.get_units_by_type(self.UNIT_TYPES['VIKING_FIGHTER'].value)

        if vikings_assault and "VikingAssault" in available_abilities:
            for viking in vikings_assault:
                actions_to_execute.append(RAW_FUNCTIONS.Morph_VikingFighterMode_quick("now", viking.tag))
                logger.info(f"Viking {viking.tag} morphing to Fighter mode")

        elif vikings_fighter and "VikingFighter" in available_abilities:
            for viking in vikings_fighter:
                actions_to_execute.append(RAW_FUNCTIONS.Morph_VikingAssaultMode_quick("now", viking.tag))
                logger.info(f"Viking {viking.tag} morphing to Assault mode")

        # 处理坦克变形
        tanks = self.get_units_by_type(self.UNIT_TYPES['SIEGE_TANK'].value)
        tanks_sieged = self.get_units_by_type(self.UNIT_TYPES['SIEGE_TANK_SIEGED'].value)

        if tanks and "SiegeTank" in available_abilities:
            for tank in tanks:
                actions_to_execute.append(RAW_FUNCTIONS.Morph_SiegeMode_quick("now", tank.tag))
                logger.info(f"Tank {tank.tag} entering Siege mode")

        elif tanks_sieged and "SiegeTankSieged" in available_abilities:
            for tank in tanks_sieged:
                actions_to_execute.append(RAW_FUNCTIONS.Morph_Unsiege_quick("now", tank.tag))
                logger.info(f"Tank {tank.tag} leaving Siege mode")

        # 处理雷神变形
        thors = self.get_units_by_type(self.UNIT_TYPES['THOR'].value)
        thors_high_impact = self.get_units_by_type(self.UNIT_TYPES['THOR_HIGH_IMPACT'].value)

        if thors and "Thor" in available_abilities:
            for thor in thors:
                actions_to_execute.append(RAW_FUNCTIONS.Morph_ThorHighImpactMode_quick("now", thor.tag))
                logger.info(f"Thor {thor.tag} entering High Impact mode")

        elif thors_high_impact and "ThorHighImpactMode" in available_abilities:
            for thor in thors_high_impact:
                actions_to_execute.append(RAW_FUNCTIONS.Morph_ThorExplosiveMode_quick("now", thor.tag))
                logger.info(f"Thor {thor.tag} returning to normal mode")

        # 处理医疗运输机技能
        medivacs = self.get_units_by_type(self.UNIT_TYPES['MEDIVAC'].value)
        if medivacs and "Medivac" in available_abilities:
            for medivac in medivacs:
                # 执行加速推进器
                if "BOOST" in available_abilities["Medivac"]:
                    actions_to_execute.append(RAW_FUNCTIONS.Effect_MedivacIgniteAfterburners_quick("now", medivac.tag))
                    logger.info(f"Medivac {medivac.tag} using boost")

                # 装载和卸载逻辑
                if medivac.cargo_space_taken == 0:  # 如果运输机是空的
                    marines = self.get_units_by_type(self.UNIT_TYPES['MARINE'].value)
                    marauders = self.get_units_by_type(self.UNIT_TYPES['MARAUDER'].value)

                    if marines:
                        actions_to_execute.append(RAW_FUNCTIONS.Load_Medivac_unit("now", medivac.tag, marines[0].tag))
                        logger.info(f"Medivac {medivac.tag} loading Marine {marines[0].tag}")
                    elif marauders:
                        actions_to_execute.append(RAW_FUNCTIONS.Load_Medivac_unit("now", medivac.tag, marauders[0].tag))
                        logger.info(f"Medivac {medivac.tag} loading Marauder {marauders[0].tag}")

                elif medivac.cargo_space_taken > 0:  # 如果运输机有装载单位
                    actions_to_execute.append(RAW_FUNCTIONS.UnloadAllAt_pt(
                        "now",
                        medivac.tag,
                        [medivac.x + 2, medivac.y]
                    ))
                    logger.info(f"Medivac {medivac.tag} unloading all units")

        return actions_to_execute

    def step(self, obs):
        super(AbilityTestAgent, self).step(obs)

        # 更新单位追踪
        self.update_unit_tracking(obs)

        # 检查技能
        if not self.abilities_checked:
            try:
                self.ability_detector.print_debug_info(obs)
                self.ability_detector.print_available_abilities(obs)
                self.abilities_checked = True
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        # 基于当前游戏状态执行动作
        actions_to_execute = self.execute_actions(obs)

        self.step_count += 1
        if self.step_count % 5 == 0:
            logger.info(f"Current step: {self.step_count}")

        return actions_to_execute
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_episode(args):
    episode, map_name = args
    agent = AbilityTestAgent()

    current_dir = os.path.abspath(os.path.dirname(__file__))
    replay_dir = os.path.join(current_dir, 'replays', f'worker_{episode}')
    ensure_dir(replay_dir)

    logger.info(f"Worker {episode} starting with map {map_name}")
    logger.info(f"Replay directory (absolute path): {replay_dir}")

    try:
        with sc2_env.SC2Env(
                map_name=map_name,
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    use_feature_units=True,
                    raw_crop_to_playable_area=True,
                    use_camera_position=True
                ),
                step_mul=1,  # 改为1以获得正常游戏速度
                game_steps_per_episode=400,  # 增加游戏时长
                visualize=True,  # 始终显示界面
        ) as env:
            logger.info(f"Worker {episode}: Environment created")

            timesteps = env.reset()
            agent.reset()

            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    # 获取结果
                    outcome = timesteps[0].reward
                    result_str = "victory" if outcome == 1 else "defeat" if outcome == -1 else "tie"

                    logger.info(f"Worker {episode}: Episode finished with result: {result_str}")
                    try:
                        save_path = env.save_replay(
                            replay_dir,
                            prefix=f"ability_test_{episode}_{result_str}_"
                        )
                        logger.info(f"Worker {episode}: Replay saved to {save_path}")
                    except Exception as e:
                        logger.error(f"Worker {episode}: Failed to save replay: {e}")
                    break
                timesteps = env.step(step_actions)

    except Exception as e:
        logger.error(f"Worker {episode}: Error during episode: {e}")


def init_worker():
    import sys
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)


def main(argv):
    logger.info("Starting main process")
    flags.FLAGS(argv)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    main_replay_dir = os.path.join(current_dir, 'replays')
    ensure_dir(main_replay_dir)
    logger.info(f"Created main replay directory (absolute path): {main_replay_dir}")

    num_episodes = 1
    map_name = "ability_test_map"
    args_list = [(i, map_name) for i in range(num_episodes)]

    with multiprocessing.Pool(num_episodes, initializer=init_worker) as pool:
        logger.info(f"Created process pool with {num_episodes} workers")
        pool.map(run_episode, args_list)

    logger.info("All episodes completed")

    # 检查replay文件
    for i in range(num_episodes):
        replay_dir = os.path.join(main_replay_dir, f'worker_{i}')
        if os.path.exists(replay_dir):
            replays = [f for f in os.listdir(replay_dir) if f.endswith('.SC2Replay')]
            logger.info(f"Worker {i} replays: {replays}")
        else:
            logger.warning(f"Replay directory for worker {i} not found")


if __name__ == "__main__":
    app.run(main)