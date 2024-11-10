import logging
import os
from typing import Optional, Tuple

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


class SC2MultimodalEnv(gym.Env):
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2  # seconds

    RESULT_MAP = {
        sc_pb.Victory: ("VICTORY", 1),
        sc_pb.Defeat: ("DEFEAT", -1),
        sc_pb.Tie: ("TIE", 0),
        sc_pb.Undecided: ("UNDECIDED", 0)
    }

    def __init__(self, map_name, save_dir, replay_dir, timestamp,
                 feature_dims=None, rgb_dims=None, map_size=None):
        super(SC2MultimodalEnv, self).__init__()

        # 验证必要的参数
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
        self.bot = None
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.game_result = None
        self.episode_count = 0
        self.total_reward = 0  # 新增:用于跟踪累积奖励
        # 设置尺寸
        self.feature_dims = tuple(feature_dims)
        self.rgb_dims = tuple(rgb_dims)
        self.map_size = tuple(map_size)

        # 从config获取颜色设置
        self.self_color = tuple(COLORS["self_color"])
        self.enemy_color = tuple(COLORS["enemy_color"])

        # 确保目录存在
        os.makedirs(self.replay_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # 设置空间
        self._define_spaces()

    def _define_spaces(self):
        """定义观察和动作空间

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

        Returns:
            None: 直接设置类的observation_space和action_space属性
        """
        # 设置单位数量上限
        self.num_units = 100

        # 定义观察空间
        unit_info_space = spaces.Sequence(
            spaces.Dict({
                # 原始tag范围设置为百万级，确保足够大
                'original_tag': spaces.Discrete(1000000),
                # 简化tag使用较小范围，便于处理
                'simplified_tag': spaces.Discrete(100),
                # alliance只有两个有效值：1(己方)和4(敌方)
                'alliance': spaces.Discrete(5),
                # unit_type范围设置为0-999
                'unit_type': spaces.Discrete(1000),
                # 单位名称最大长度50
                'unit_name': spaces.Text(max_length=50),
                # 生命值、护盾值、能量值均为非负浮点数
                'health': spaces.Box(low=0, high=float('inf'), shape=()),
                'max_health': spaces.Box(low=0, high=float('inf'), shape=()),
                'shield': spaces.Box(low=0, high=float('inf'), shape=()),
                'max_shield': spaces.Box(low=0, high=float('inf'), shape=()),
                'energy': spaces.Box(low=0, high=float('inf'), shape=()),
                # 位置坐标为二维非负浮点数
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
            # 攻击者和目标的simplified_tag范围为[0, num_units-1]
            spaces.Discrete(self.num_units),  # attacker_tag
            spaces.Discrete(self.num_units)  # target_tag
        ))

        move_space = spaces.Tuple((
            spaces.Discrete(3),  # move_type: 0(no_move), 1(grid_move), 2(smac_move)
            spaces.Discrete(self.num_units),  # unit_tag
            spaces.Box(  # target position or direction
                low=0,
                high=10,  # 改为10而不是9，因为是左闭右开区间[0,10)
                shape=(2,),
                dtype=np.int32
            )
        ))

        self.action_space = spaces.Dict({
            'attack': attack_space,
            'move': move_space
        })

    def _get_game_result(self, timestep) -> Tuple[str, int]:
        """
        从timestep获取游戏结果
        返回格式: (结果字符串, 结果数值)
        """
        try:
            # 如果不是最后一步，返回未决定状态
            if not timestep.last():
                return ("UNDECIDED", 0)

            # 获取原始观察值
            raw_obs = self.sc2_env._obs[0]  # 直接获取原始观察值
            outcome = 0  # 默认值

            # 检查是否有game_loop(确保是有效的观察值)
            if hasattr(raw_obs.observation, 'game_loop'):
                # 从player_result获取结果
                if raw_obs.player_result:
                    # 获取当前玩家ID
                    player_id = raw_obs.observation.player_common.player_id
                    # 遍历结果找到对应玩家的结果
                    for result in raw_obs.player_result:
                        if result.player_id == player_id:
                            possible_results = {
                                sc_pb.Victory: ("VICTORY", 1),
                                sc_pb.Defeat: ("DEFEAT", -1),
                                sc_pb.Tie: ("TIE", 0),
                                sc_pb.Undecided: ("UNDECIDED", 0),
                            }
                            result_tuple = possible_results.get(result.result, ("UNDECIDED", 0))
                            logger.info(f"Game ended with result: {result_tuple[0]}")
                            return result_tuple

            # 如果无法从player_result获取结果，检查observation中的outcome
            if hasattr(timestep, 'observation') and 'score_cumulative' in timestep.observation:
                outcome_value = timestep.observation.get('outcome', [0])[0]
                if outcome_value > 0:
                    logger.info("Game ended with VICTORY from observation outcome")
                    return ("VICTORY", 1)
                elif outcome_value < 0:
                    logger.info("Game ended with DEFEAT from observation outcome")
                    return ("DEFEAT", -1)

            # 最后检查reward
            reward = timestep.reward
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0]

            logger.info(f"Determining result from reward: {reward}")
            if reward > 0:
                return ("VICTORY", 1)
            elif reward < 0:
                return ("DEFEAT", -1)

            return ("TIE", 0)

        except Exception as e:
            logger.error(f"Error getting game result: {str(e)}", exc_info=True)
            return ("UNDECIDED", 0)




    def _create_env(self):
        """创建SC2环境"""
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
            # 使用基础文件名，让sc2_env生成初始replay
            return sc2_env.SC2Env(
                map_name=self.map_name,
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
                agent_interface_format=agent_interface_format,
                step_mul=8,
                game_steps_per_episode=1000,
                visualize=True,
                score_index=0,  # 使用dense reward
                score_multiplier=1,  # 可以调整reward的scale
            )
        except Exception as e:
            logger.error(f"Failed to create SC2Env: {e}")
            raise

    def _save_replay_with_result(self, result_tuple: Tuple[str, int]):
        """使用游戏结果和累积奖励保存回放"""
        result_str, result_value = result_tuple
        try:
            if self.sc2_env:
                # 构建完整的replay名称,包含结果信息
                prefix = (f"{self.timestamp}_episode_{self.episode_count:04d}_"
                          f"{result_str.lower()}_reward_{self.total_reward:.2f}_")  # 使用小写的结果字符串

                save_path = self.sc2_env.save_replay(
                    self.replay_dir,
                    prefix=prefix
                )
                logger.info(f"Replay saved as: {save_path}")
        except Exception as e:
            logger.error(f"Error saving replay: {str(e)}")

    def step(self, action):
        """单步执行"""
        try:
            # 设置机器人的动作命令
            try:
                self.bot.attack_commands = action.get('attack', [])
                self.bot.original_move_commands = []
                self.bot.smac_move_commands = []
                for move_command in action.get('move', []):
                    if len(move_command) == 3:
                        move_type, unit_index, move_target = move_command
                        if move_type == 1:  # 原始移动
                            self.bot.original_move_commands.append((unit_index, tuple(move_target)))
                        elif move_type == 2:  # SMAC移动
                            self.bot.smac_move_commands.append((unit_index, move_target[0]))
            except Exception as e:
                return None, 0, True, {"error": f"Bot command setting error: {str(e)}"}

            # 1. 执行空步骤并获取观察
            obs = self.sc2_env.step([])

            # 如果已经是最后一步，直接处理结果
            if obs[0].last():
                next_state = self._get_observation(obs[0])
                result_str, result_value = self._get_game_result(obs[0])
                reward = result_value
                self.total_reward += reward  # 累积最终奖励
                info = {
                    'game_result': result_str,
                    'game_result_value': result_value,
                    'total_reward': self.total_reward
                }
                self._save_replay_with_result((result_str, result_value))
                return next_state, result_value, True, info

            # 2. 获取机器人动作
            try:
                bot_actions = self.bot.step(obs[0])
            except Exception as e:
                return None, 0, True, {"error": f"Bot action error: {str(e)}"}

            # 3. 执行动作
            obs = self.sc2_env.step([bot_actions])
            next_state = self._get_observation(obs[0])
            done = obs[0].last()

            if done:
                result_str, result_value = self._get_game_result(obs[0])
                reward = result_value
                self.total_reward += reward  # 累积最终奖励
                info = {
                    'game_result': result_str,
                    'game_result_value': result_value,
                    'total_reward': self.total_reward
                }
                self._save_replay_with_result((result_str, result_value))
            else:
                reward = 0
                self.total_reward += reward  # 累积过程中的奖励(这里都是0)
                info = {'total_reward': self.total_reward}

            return next_state, reward, done, info

        except Exception as e:
            logger.error(f"Error in step method: {str(e)}")
            return None, 0, True, {"error": str(e)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        try:
            # 关闭现有环境
            if self.sc2_env is not None:
                self.close()
            self.episode_count += 1
            self.total_reward = 0  # 重置累积奖励
            logger.info(f"\nStarting Episode {self.episode_count:04d}")

            # 创建新环境
            self.sc2_env = self._create_env()
            obs = self.sc2_env.reset()

            # 重置机器人
            self.bot = Multimodal_bot(
                self_color=self.self_color,
                enemy_color=self.enemy_color,
                feature_dims=self.feature_dims,
                rgb_dims=self.rgb_dims,
                map_size=self.map_size
            )
            self.bot.setup(self.sc2_env.observation_spec(), self.sc2_env.action_spec())
            self.bot.reset()

            # 重置状态
            self.previous_score = None
            return self._get_observation(obs[0])

        except Exception as e:
            logger.error(f"Error during reset: {e}", exc_info=True)
            self.close()
            raise

    def close(self):
        """关闭环境"""
        if self.sc2_env:
            self.sc2_env.close()
            self.sc2_env = None

    def __del__(self):
        self.close()

    def _get_observation(self, obs):
        """获取观察信息"""
        try:
            raw_image, unit_info = self.bot.get_raw_image_and_unit_info(obs)
            text_description = self.bot.get_text_description(obs)

            # 处理单位信息，按alliance和unit_type分组进行编号
            processed_unit_info = []

            # 首先按simplified_tag排序，确保编号顺序一致
            sorted_units = sorted(unit_info, key=lambda x: x['simplified_tag'])

            # 用于跟踪当前处理的单位类型
            type_alliance_count = {}

            for unit in sorted_units:
                # 获取基础单位名称
                base_unit_name = get_unit_name(unit['unit_type'])

                # 使用(unit_type, alliance)作为计数器的键
                counter_key = (unit['unit_type'], unit['alliance'])

                # 获取并更新该类型的计数
                if counter_key not in type_alliance_count:
                    type_alliance_count[counter_key] = 1
                else:
                    type_alliance_count[counter_key] += 1

                # 生成单位名称，使用simplified_tag确保稳定性
                numbered_unit_name = f"{base_unit_name}_{type_alliance_count[counter_key]}"

                unit_data = {
                    'original_tag': unit['original_tag'],
                    'simplified_tag': unit['simplified_tag'],
                    'alliance': unit['alliance'],
                    'unit_type': unit['unit_type'],
                    'unit_name': numbered_unit_name,
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
            logger.error(f"Error in _get_observation: {e}")
            raise






