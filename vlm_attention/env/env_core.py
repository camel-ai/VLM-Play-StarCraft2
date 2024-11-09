import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from threading import RLock
from pysc2.env import sc2_env
from pysc2.lib import actions, features, protocol
from vlm_attention.env.env_bot import Multimodal_bot
from vlm_attention.env.config import COLORS, get_unit_name
from s2clientprotocol import sc2api_pb2 as sc_pb
import logging
from contextlib import contextmanager
import os
import time
import websocket
import psutil

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

    def __init__(self, map_name, save_dir, replay_dir, timestamp, save_replay_episodes=1,
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
        self.save_replay_episodes = save_replay_episodes
        self.env_lock = RLock()
        self.sc2_env = None
        self.bot = None
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.game_result = None
        self.episode_count = 0

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
        """
        try:
            # 检查是否是最后一步
            if not timestep.last():
                return ("UNDECIDED", 0)

            # 从observation中获取player_result
            if hasattr(timestep, 'observation') and 'player_result' in timestep.observation:
                player_result = timestep.observation['player_result']
                if player_result:
                    result = player_result[0].result
                    return self.RESULT_MAP.get(result, ("UNDECIDED", 0))

            # 如果上面的方法失败，尝试从reward判断
            if hasattr(timestep, 'reward'):
                reward = timestep.reward
                if reward == 1:
                    return ("VICTORY", 1)
                elif reward == -1:
                    return ("DEFEAT", -1)

            logger.warning("Unable to determine game result from timestep")
            return ("UNDECIDED", 0)

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
                save_replay_episodes=self.save_replay_episodes,
                replay_dir=self.replay_dir,
                replay_prefix=f"{self.timestamp}_episode_{self.episode_count:04d}",  # 简化前缀
                score_index=-1
            )
        except Exception as e:
            logger.error(f"Failed to create SC2Env: {e}")
            raise

    def _save_replay_with_result(self, result_tuple: Tuple[str, int]):
        """
        使用游戏结果保存回放文件
        """
        result_str, result_value = result_tuple
        try:
            base_filename = f"{self.timestamp}_episode_{self.episode_count:04d}"

            # 等待一小段时间确保replay文件已经生成
            max_attempts = 5
            attempt = 0
            while attempt < max_attempts:
                replay_files = [f for f in os.listdir(self.replay_dir)
                                if f.startswith(base_filename)]

                if replay_files:
                    latest_replay = max(replay_files, key=lambda f: os.path.getctime(
                        os.path.join(self.replay_dir, f)))

                    # 构建新的文件名，包含结果
                    new_name = (f"{base_filename}_"
                                f"result_{result_str}"
                                f".SC2Replay")

                    old_path = os.path.join(self.replay_dir, latest_replay)
                    new_path = os.path.join(self.replay_dir, new_name)

                    # 确保旧文件存在且可访问
                    if os.path.exists(old_path):
                        # 如果新文件已存在，先删除
                        if os.path.exists(new_path):
                            os.remove(new_path)

                        # 重命名文件
                        os.rename(old_path, new_path)
                        logger.info(f"Replay saved as: {new_name}")
                        return

                attempt += 1
                time.sleep(0.5)  # 等待500ms再次尝试

            logger.warning(f"Failed to save replay for episode {self.episode_count} after {max_attempts} attempts")

        except Exception as e:
            logger.error(f"Error saving replay with result: {e}")
    def _terminate_sc2_processes(self):
        """确保终止所有SC2相关进程"""
        for proc in psutil.process_iter(['name']):
            try:
                if 'SC2' in proc.info['name']:
                    proc.kill()
                    proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired, Exception) as e:
                logger.error(f"Error terminating SC2 process: {e}")
        time.sleep(1)  # 等待进程完全终止

    def step(self, action):
        """单步执行"""
        try:
            with self._env_context() as env:
                # 设置机器人的动作命令
                try:
                    # 设置攻击命令
                    self.bot.attack_commands = action.get('attack', [])

                    # 设置移动命令
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

                # 1. 第一阶段：执行空步骤并获取观察
                try:
                    obs = env.step([])
                except (protocol.ConnectionError, websocket.WebSocketConnectionClosedException):
                    # 如果是连接错误，返回终止状态
                    return None, 0, True, {"error": "Connection lost during initial step"}

                # 如果已经是最后一步，直接处理结果
                if obs[0].last():
                    next_state = self._get_observation(obs[0])
                    result_str, result_value = self._get_game_result(obs[0])
                    info = {
                        'game_result': result_str,
                        'game_result_value': result_value,
                        'final_score': 0
                    }
                    self._save_replay_with_result((result_str, result_value))
                    return next_state, result_value, True, info

                # 2. 第二阶段：获取机器人动作
                try:
                    bot_actions = self.bot.step(obs[0])
                except Exception as e:
                    return None, 0, True, {"error": f"Bot action error: {str(e)}"}

                # 3. 第三阶段：执行动作
                try:
                    obs = env.step([bot_actions])
                    next_state = self._get_observation(obs[0])
                    done = obs[0].last()

                    if done:
                        result_str, result_value = self._get_game_result(obs[0])
                        reward = result_value
                        info = {
                            'game_result': result_str,
                            'game_result_value': result_value,
                            'final_score': 0
                        }
                        self._save_replay_with_result((result_str, result_value))
                    else:
                        reward = 0
                        info = {}

                    return next_state, reward, done, info

                except (protocol.ConnectionError, websocket.WebSocketConnectionClosedException):
                    # 如果在执行动作时发生连接错误，返回终止状态
                    return None, 0, True, {"error": "Connection lost during action execution"}

        except Exception as e:
            logger.error(f"Error in step method: {str(e)}", exc_info=True)
            return None, 0, True, {"error": str(e)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        try:
            # 关闭现有环境
            if self.sc2_env is not None:
                self.close()

            self.episode_count += 1
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
        """确保正确关闭环境"""
        with self.env_lock:
            if self.sc2_env:
                try:
                    self.sc2_env.close()
                    self.sc2_env = None
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Error while closing SC2Env: {e}")
                finally:
                    self._terminate_sc2_processes()
                    time.sleep(1)

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

    def _compute_reward(self, obs):
        """
        计算奖励 - 当score_index=-1时，直接使用胜负作为奖励
        胜利返回1，失败返回-1，其他情况返回0
        """
        try:
            # 如果游戏结束，返回最终结果奖励
            if obs.last():
                _, result_value = self._get_game_result(obs)
                return result_value
            return 0  # 游戏进行中返回0

        except Exception as e:
            logger.error(f"Error in _compute_reward: {e}")
            return 0

    def _handle_connection_error(self, error: Exception) -> bool:
        """
        处理连接错误，决定是否需要重试
        返回: True 如果应该重试，False 如果应该放弃
        """
        current_time = time.time()
        if current_time - self.last_error_time > 60:  # 重置计数器如果距离上次错误超过60秒
            self.consecutive_errors = 0

        self.consecutive_errors += 1
        self.last_error_time = current_time

        if self.consecutive_errors >= self.MAX_RETRY_ATTEMPTS:
            logger.error(f"Too many consecutive errors ({self.consecutive_errors}), giving up")
            return False

        logger.warning(f"Connection error occurred (attempt {self.consecutive_errors}/{self.MAX_RETRY_ATTEMPTS}). "
                       f"Retrying in {self.RETRY_DELAY} seconds...")
        time.sleep(self.RETRY_DELAY)
        return True

    @contextmanager
    def _env_context(self):
        """环境上下文管理器"""
        max_retries = 3
        retry_delay = 2
        current_retry = 0

        while current_retry < max_retries:
            try:
                with self.env_lock:
                    if self.sc2_env is None:
                        self.sc2_env = self._create_env()
                    yield self.sc2_env
                    break

            except (protocol.ConnectionError, websocket.WebSocketConnectionClosedException) as e:
                current_retry += 1
                logger.warning(f"Connection error (attempt {current_retry}/{max_retries}): {str(e)}")

                # 清理当前环境
                try:
                    if self.sc2_env:
                        self.sc2_env.close()
                        self.sc2_env = None
                except:
                    pass

                # 最后一次重试失败就抛出异常
                if current_retry >= max_retries:
                    raise

                # 重试前等待更长时间
                time.sleep(retry_delay * current_retry)

                # 确保SC2进程被清理
                self._terminate_sc2_processes()
                time.sleep(1)

            except Exception as e:
                logger.error(f"Unexpected error in env_context: {str(e)}")
                raise

