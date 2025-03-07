import logging
import os
from typing import Optional, Tuple, List

import gym
import numpy as np
from gym import spaces
from s2clientprotocol import sc2api_pb2 as sc_pb

from pysc2.env import sc2_env
from pysc2.lib import actions, features
from vlm_attention.env.config import COLORS, get_unit_name
from vlm_attention.env.env_bot import Multimodal_bot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
environment with two players, without ability support

this environment provide an easier interface for building the environment with two players without ability support

based on this environment, we can directly interact with agent through out text and image
"""
class SC2MultimodalTwoPlayerEnv(gym.Env):
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2

    RESULT_MAP = {
        sc_pb.Victory: ("VICTORY", 1),
        sc_pb.Defeat: ("DEFEAT", -1),
        sc_pb.Tie: ("TIE", 0),
        sc_pb.Undecided: ("UNDECIDED", 0)
    }

    def __init__(self, map_name, save_dir, replay_dir, timestamp,
                 feature_dims=None, rgb_dims=None, map_size=None):
        super(SC2MultimodalTwoPlayerEnv, self).__init__()

        # 验证参数
        if feature_dims is None or rgb_dims is None or map_size is None:
            raise ValueError("feature_dims, rgb_dims, and map_size must be provided")

        if not isinstance(feature_dims, (tuple, list)) or len(feature_dims) != 2:
            raise ValueError("feature_dims must be a tuple/list of length 2")

        if not isinstance(rgb_dims, (tuple, list)) or len(rgb_dims) != 2:
            raise ValueError("rgb_dims must be a tuple/list of length 2")

        if not isinstance(map_size, (tuple, list)) or len(map_size) != 2:
            raise ValueError("map_size must be a tuple/list of length 2")

        self.map_name = map_name
        self.save_dir = save_dir
        self.replay_dir = replay_dir
        self.timestamp = timestamp
        self.sc2_env = None
        self.bots = [None, None]  # 两个玩家的bot
        self.episode_count = 0
        self.total_rewards = [0, 0]  # 两个玩家的累积奖励

        # 设置尺寸
        self.feature_dims = tuple(feature_dims)
        self.rgb_dims = tuple(rgb_dims)
        self.map_size = tuple(map_size)

        # 颜色设置
        self.self_color = tuple(COLORS["self_color"])
        self.enemy_color = tuple(COLORS["enemy_color"])

        # 确保目录存在
        os.makedirs(self.replay_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # 设置空间
        self._define_spaces()

    def _define_spaces(self):
        """定义观察和动作空间，与单玩家版本相同

        Observation Space 包含三个主要部分:
        1. Text Observation
           - 游戏状态的文字描述
           - 包含所有单位的生命值、护盾值等状态信息
           - 按阵营(己方/敌方)分组组织

        2. Image Observation
           - RGB格式的游戏画面截图
           - 形状: (height, width, 3)
           - 像素值范围: [0, 255]
           - 包含所有单位的可视化表示

        3. Unit Information
           - original_tag: PySC2原生单位标识符 (0 ~ 999999)
           - simplified_tag: 简化的单位标识符 (1 ~ 99)
           - alliance: 单位阵营 (1:己方, 4:敌方)
           - unit_type: 单位类型编号 (0 ~ 999)
           - unit_name: 单位类型名称
           - health/shield/energy: 单位状态值
           - position: 单位在屏幕上的坐标 (x, y)

        Action Space 包含两种动作类型:
        1. Attack Actions
           Format: (attacker_tag, target_tag)
           - attacker_tag: 发起攻击的己方单位的simplified_tag (0 ~ num_units-1)
           - target_tag: 攻击目标的simplified_tag (0 ~ num_units-1)
           注意: 验证时需确保attacker是己方单位，target是敌方单位

        2. Move Actions
           Format: (move_type, unit_tag, target)

          2.1 Grid-based Movement (move_type = 1)
               - unit_tag: 移动单位的simplified_tag (0 ~ num_units-1)
               - target: 目标网格坐标 [x, y], x,y均在[0, 9]范围内
               坐标系说明:
               - 使用10x10网格
               - 原点(0,0)在左上角
               - x轴向右为正，范围[0,9]
               - y轴向下为正，范围[0,9]

           2.2 SMAC-style Movement (move_type = 2)
               - unit_tag: 移动单位的simplified_tag (0 ~ num_units-1)
               - target: [direction, 0]
               direction说明:
               - 0: 向上移动(y-1)
               - 1: 向右移动(x+1)
               - 2: 向下移动(y+1)
               - 3: 向左移动(x-1)

        Action Examples (两个玩家):
        -----------------------
        1. 玩家1和玩家2的动作列表:
           actions = [player1_action, player2_action]

        2. 玩家1攻击示例:
           player1_action = {'attack': (1, 6), 'move': (0, 0, [0, 0])}
           
        3. 玩家2移动示例:
           player2_action = {'attack': [], 'move': (1, 3, [5, 7])}
           
        4. 完整示例 - 玩家1攻击，玩家2移动:
           actions = [
               {'attack': (1, 6), 'move': (0, 0, [0, 0])},  # 玩家1
               {'attack': [], 'move': (1, 3, [5, 7])}       # 玩家2
           ]
           
        5. 完整示例 - 两个玩家都攻击:
           actions = [
               {'attack': (1, 8), 'move': (0, 0, [0, 0])},  # 玩家1
               {'attack': (2, 7), 'move': (0, 0, [0, 0])}   # 玩家2
           ]
           
        6. 完整示例 - 两个玩家都移动:
           actions = [
               {'attack': [], 'move': (1, 2, [3, 4])},      # 玩家1
               {'attack': [], 'move': (2, 1, [0, 0])}       # 玩家2 (SMAC向上移动)
           ]
           
        7. 完整示例 - 多个攻击命令:
           actions = [
               {'attack': [(1, 6), (2, 8)], 'move': (0, 0, [0, 0])},  # 玩家1
               {'attack': [(3, 5), (4, 7)], 'move': (0, 0, [0, 0])}   # 玩家2
           ]
        """
        self.num_units = 100

        # 定义观察空间
        unit_info_space = spaces.Sequence(
            spaces.Dict({
                'original_tag': spaces.Discrete(1000000),
                'simplified_tag': spaces.Discrete(100),
                'alliance': spaces.Discrete(5),
                'unit_type': spaces.Discrete(1000),
                'unit_name': spaces.Text(max_length=50),
                'health': spaces.Box(low=0, high=float('inf'), shape=()),
                'max_health': spaces.Box(low=0, high=float('inf'), shape=()),
                'shield': spaces.Box(low=0, high=float('inf'), shape=()),
                'max_shield': spaces.Box(low=0, high=float('inf'), shape=()),
                'energy': spaces.Box(low=0, high=float('inf'), shape=()),
                'position': spaces.Box(low=0, high=float('inf'), shape=(2,))
            })
        )

        self.observation_space = spaces.Dict({
            'text': spaces.Text(max_length=1000),
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(*self.rgb_dims, 3),
                dtype=np.uint8
            ),
            'unit_info': unit_info_space
        })

        # 定义动作空间
        attack_space = spaces.Tuple((
            spaces.Discrete(self.num_units),
            spaces.Discrete(self.num_units)
        ))

        move_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Discrete(self.num_units),
            spaces.Box(
                low=0,
                high=10,
                shape=(2,),
                dtype=np.int32
            )
        ))

        self.action_space = spaces.Dict({
            'attack': attack_space,
            'move': move_space
        })

    def _create_env(self):
        """创建支持两个玩家的SC2环境"""
        agent_interface_format = features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=self.feature_dims, minimap=(32, 32)),
            rgb_dimensions=features.Dimensions(screen=self.rgb_dims, minimap=(64, 64)),
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
        )

        try:
            return sc2_env.SC2Env(
                map_name=self.map_name,
                players=[
                    sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Agent(sc2_env.Race.terran)  # 两个玩家都是Agent
                ],
                agent_interface_format=agent_interface_format,
                step_mul=8,
                game_steps_per_episode=1000,
                visualize=True,
                score_index=0,  # 禁用默认的得分计算
                score_multiplier=1
            )
        except Exception as e:
            logger.error(f"Failed to create SC2Env: {e}")
            raise

    def _get_game_result(self, timesteps) -> List[Tuple[str, int]]:
        """获取两个玩家的游戏结果

        Args:
            timesteps: 包含两个玩家观察值的列表

        Returns:
            List[Tuple[str, int]]: 两个玩家的结果，每个元素为(结果字符串, 结果数值)
        """
        try:
            if not timesteps[0].last():
                return [("UNDECIDED", 0), ("UNDECIDED", 0)]

            results = []
            for i, timestep in enumerate(timesteps):
                raw_obs = self.sc2_env._obs[i]

                if hasattr(raw_obs.observation, 'game_loop'):
                    if raw_obs.player_result:
                        player_id = raw_obs.observation.player_common.player_id
                        for result in raw_obs.player_result:
                            if result.player_id == player_id:
                                result_tuple = self.RESULT_MAP.get(result.result, ("UNDECIDED", 0))
                                results.append(result_tuple)
                                break

                if len(results) <= i:  # 如果还没有结果，检查其他指标
                    reward = timestep.reward
                    if isinstance(reward, (list, np.ndarray)):
                        reward = reward[0]

                    if reward > 0:
                        results.append(("VICTORY", 1))
                    elif reward < 0:
                        results.append(("DEFEAT", -1))
                    else:
                        results.append(("TIE", 0))

            # 确保返回两个结果
            while len(results) < 2:
                results.append(("UNDECIDED", 0))

            # 记录结果
            logger.info(f"Game results - Player 1: {results[0][0]}, Player 2: {results[1][0]}")
            return results

        except Exception as e:
            logger.error(f"Error getting game results: {str(e)}", exc_info=True)
            return [("UNDECIDED", 0), ("UNDECIDED", 0)]

    def _get_observation(self, obs, player_idx):
        """获取指定玩家的观察信息

        Args:
            obs: 当前观察
            player_idx: 玩家索引(0或1)
        """
        try:
            if not self.bots[player_idx]:
                raise ValueError(f"Bot for player {player_idx} is not initialized")

            bot = self.bots[player_idx]
            raw_image, unit_info = bot.get_raw_image_and_unit_info(obs)
            text_description = bot.get_text_description(obs)

            processed_unit_info = []
            sorted_units = sorted(unit_info, key=lambda x: x['simplified_tag'])

            for unit in sorted_units:
                original_tag = unit['original_tag']
                _, unit_name = bot.unit_manager.tag_registry.get(original_tag, (None, None))

                if unit_name is None:
                    logger.warning(f"Unit with original_tag {original_tag} not found in tag_registry")
                    continue

                unit_data = {
                    'original_tag': original_tag,
                    'simplified_tag': unit['simplified_tag'],
                    'alliance': unit['alliance'],
                    'unit_type': unit['unit_type'],
                    'unit_name': unit_name,
                    'health': unit['health'],
                    'max_health': unit['max_health'],
                    'shield': unit['shield'],
                    'max_shield': unit['max_shield'],
                    'energy': unit['energy'],
                    'position': unit['position']
                }
                processed_unit_info.append(unit_data)

            return {
                'text': text_description,
                'image': raw_image,
                'unit_info': processed_unit_info
            }
        except Exception as e:
            logger.error(f"Error in _get_observation for player {player_idx}: {e}")
            raise

    def _save_replay_with_result(self, results: List[Tuple[str, int]]):
        """保存带有两个玩家结果的回放"""
        try:
            if self.sc2_env:
                result_str = f"p1_{results[0][0].lower()}_p2_{results[1][0].lower()}"
                total_reward_str = f"r1_{self.total_rewards[0]:.2f}_r2_{self.total_rewards[1]:.2f}"
                prefix = f"{self.timestamp}_episode_{self.episode_count:04d}_{result_str}_{total_reward_str}_"

                save_path = self.sc2_env.save_replay(
                    self.replay_dir,
                    prefix=prefix
                )
                logger.info(f"Replay saved as: {save_path}")
        except Exception as e:
            logger.error(f"Error saving replay: {str(e)}")

    def step(self, actions):
        """执行两个玩家的动作

        Args:
            actions: 包含两个玩家动作的列表
        """
        try:
            # 设置两个bot的动作命令
            for i, (bot, action) in enumerate(zip(self.bots, actions)):
                try:
                    bot.attack_commands = action.get('attack', [])
                    bot.original_move_commands = []
                    bot.smac_move_commands = []
                    for move_command in action.get('move', []):
                        if len(move_command) == 3:
                            move_type, unit_index, move_target = move_command
                            if move_type == 1:
                                bot.original_move_commands.append((unit_index, tuple(move_target)))
                            elif move_type == 2:
                                bot.smac_move_commands.append((unit_index, move_target[0]))
                except Exception as e:
                    return None, [0, 0], True, {"error": f"Bot {i} command setting error: {str(e)}"}

            # 1. 执行空步骤
            timesteps = self.sc2_env.step([])

            # 检查是否是最后一步
            if timesteps[0].last():
                observations = [self._get_observation(ts, i) for i, ts in enumerate(timesteps)]
                results = self._get_game_result(timesteps)
                rewards = [result[1] for result in results]
                self.total_rewards = [total + r for total, r in zip(self.total_rewards, rewards)]
                info = {
                    'game_results': [result[0] for result in results],
                    'game_result_values': rewards,
                    'total_rewards': self.total_rewards
                }
                self._save_replay_with_result(results)
                return observations, rewards, True, info

            # 2. 获取bot动作
            try:
                bot_actions = [bot.step(timestep) for bot, timestep in zip(self.bots, timesteps)]
            except Exception as e:
                return None, [0, 0], True, {"error": f"Bot action error: {str(e)}"}

            # 3. 执行动作
            timesteps = self.sc2_env.step(bot_actions)
            observations = [self._get_observation(ts, i) for i, ts in enumerate(timesteps)]
            done = timesteps[0].last()

            if done:
                results = self._get_game_result(timesteps)
                rewards = [result[1] for result in results]
                self.total_rewards = [total + r for total, r in zip(self.total_rewards, rewards)]
                info = {
                    'game_results': [result[0] for result in results],
                    'game_result_values': rewards,
                    'total_rewards': self.total_rewards
                }
                self._save_replay_with_result(results)
            else:
                rewards = [0, 0]
                self.total_rewards = [total + r for total, r in zip(self.total_rewards, rewards)]
                info = {'total_rewards': self.total_rewards}

            return observations, rewards, done, info

        except Exception as e:
            logger.error(f"Error in step method: {str(e)}")
            return None, [0, 0], True, {"error": str(e)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境，初始化两个玩家的bot"""
        try:
            if self.sc2_env is not None:
                self.close()

            self.episode_count += 1
            self.total_rewards = [0, 0]
            logger.info(f"\nStarting Episode {self.episode_count:04d}")

            # 创建新环境
            self.sc2_env = self._create_env()
            timesteps = self.sc2_env.reset()

            # 创建两个bot实例
            self.bots = [
                Multimodal_bot(
                    self_color=self.self_color,
                    enemy_color=self.enemy_color,
                    feature_dims=self.feature_dims,
                    rgb_dims=self.rgb_dims,
                    map_size=self.map_size
                ) for _ in range(2)
            ]

            # 设置两个bot
            for bot in self.bots:
                bot.setup(self.sc2_env.observation_spec(), self.sc2_env.action_spec())
                bot.reset()

                # 重置UnitManager状态
                bot.unit_manager.initialized = False
                bot.unit_manager.tag_registry.clear()
                bot.unit_manager.type_counters.clear()
                bot.unit_manager.next_simplified_tag = 1

            # 返回两个玩家的初始观察
            return [self._get_observation(timestep, i) for i, timestep in enumerate(timesteps)]

        except Exception as e:
            logger.error(f"Error during reset: {e}", exc_info=True)
            self.close()
            raise

    def close(self):
        """关闭环境"""
        if self.sc2_env:
            self.sc2_env.close()
            self.sc2_env = None
        self.bots = [None, None]

    def __del__(self):
        self.close()