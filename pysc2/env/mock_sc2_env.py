# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mocking the Starcraft II environment."""

import numpy as np
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import units
from pysc2.tests import dummy_observation

from s2clientprotocol import common_pb2
from s2clientprotocol import raw_pb2
from s2clientprotocol import sc2api_pb2

DUMMY_MAP_SIZE = 256


class _TestEnvironment(environment.Base):
  """A simple generic test environment.

  This class is a lightweight implementation of `environment.Base` that returns
  the same timesteps on every observation call. By default, each returned
  timestep (one per agent) is reward 0., discount 1., and the observations are
  zero `np.ndarrays` of dtype `np.int32` and the shape specified by the
  environment's spec.

  However, the behavior of the `TestEnvironment` can be configured using the
  object's attributes.

  Attributes:
    next_timestep: The `environment.TimeStep`s to return on the next call to
      `step`. When necessary, some fields will be overridden to ensure the
      `step_type` contract.
    episode_length: if the episode length (number of transitions) exceeds
      `episode_length` on a call to `step`, the `step-type` will be set to
      `environment.StepType.LAST`, forcing an end of episode. This allows a
      stub of a production environment to have end_episodes. Will be ignored if
      set to `float('inf')` (the default).
  """

  def __init__(self, num_agents, observation_spec, action_spec):
    """Initializes the TestEnvironment.

    The `next_observation` is initialized to be reward = 0., discount = 1.,
    and an appropriately sized observation of all zeros. `episode_length` is set
    to `float('inf')`.

    Args:
      num_agents: The number of agents.
      observation_spec: The observation specs for each player.
      action_spec: The action specs for each player.
    """
    self._num_agents = num_agents
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._episode_steps = 0

    self.next_timestep = []
    for agent_index, obs_spec in enumerate(observation_spec):
      self.next_timestep.append(environment.TimeStep(
          step_type=environment.StepType.MID,
          reward=0.,
          discount=1.,
          observation=self._default_observation(obs_spec, agent_index)))

    self.episode_length = float('inf')

  def reset(self):
    """Restarts episode and returns `next_observation` with `StepType.FIRST`."""
    self._episode_steps = 0
    return self.step([None] * self._num_agents)

  def step(self, actions, step_mul=None):
    """Returns `next_observation` modifying its `step_type` if necessary."""
    del step_mul  # ignored currently

    if len(actions) != self._num_agents:
      raise ValueError(
          'Expected %d actions, received %d.' % (
              self._num_agents, len(actions)))

    if self._episode_steps == 0:
      step_type = environment.StepType.FIRST
    elif self._episode_steps >= self.episode_length:
      step_type = environment.StepType.LAST
    else:
      step_type = environment.StepType.MID

    timesteps = []
    for timestep in self.next_timestep:
      if step_type is environment.StepType.FIRST:
        timesteps.append(timestep._replace(
            step_type=step_type,
            reward=0.,
            discount=0.))
      elif step_type is environment.StepType.LAST:
        timesteps.append(timestep._replace(
            step_type=step_type,
            discount=0.))
      else:
        timesteps.append(timestep)

    if timesteps[0].step_type is environment.StepType.LAST:
      self._episode_steps = 0
    else:
      self._episode_steps += 1

    return timesteps

  def action_spec(self):
    """See base class."""
    return self._action_spec

  def observation_spec(self):
    """See base class."""
    return self._observation_spec

  def _default_observation(self, obs_spec, agent_index):
    """Returns an observation based on the observation spec."""
    observation = {}
    for key, spec in obs_spec.items():
      observation[key] = np.zeros(shape=spec, dtype=np.int32)
    return observation


class SC2TestEnv(_TestEnvironment):
  """A TestEnvironment to swap in for `starcraft2.env.sc2_env.SC2Env`.

  Repeatedly returns a mock observation for 10 calls to `step` whereupon it
  sets discount to 0. and changes state to READY_TO_END_EPISODE.

  Example:

  ```
  @mock.patch(
      'starcraft2.env.sc2_env.SC2Env',
      mock_sc2_env.SC2TestEnv)
  def test_method(self):
    env = sc2_env.SC2Env('nonexisting map')  # Really a SC2TestEnv.
    ...
  ```

  See base class for more details.
  """

  def __init__(self,
               *,
               map_name=None,
               players=None,
               agent_interface_format=None,
               discount=1.,
               discount_zero_after_timeout=False,
               visualize=False,
               step_mul=None,
               realtime=False,
               save_replay_episodes=0,
               replay_dir=None,
               game_steps_per_episode=None,
               score_index=None,
               score_multiplier=None,
               random_seed=None,
               disable_fog=False,
               ensure_available_actions=True,
               version=None):
    """Initializes an SC2TestEnv.

    Args:
      map_name: Map name. Ignored.
      players: A list of Agent and Bot instances that specify who will play.
      agent_interface_format: A sequence containing one AgentInterfaceFormat per
        agent, matching the order of agents specified in the players list. Or
        a single AgentInterfaceFormat to be used for all agents. Note that
        InterfaceOptions may be supplied in place of AgentInterfaceFormat, in
        which case no action or observation processing will be carried out by
        PySC2. The sc_pb.ResponseObservation proto will be returned as the
        observation for the agent and passed actions must be instances of
        sc_pb.Action. This is intended for agents which use custom environment
        conversion code.
      discount: Unused.
      discount_zero_after_timeout: Unused.
      visualize: Unused.
      step_mul: Unused.
      realtime: Not supported by the mock environment, throws if set to true.
      save_replay_episodes: Unused.
      replay_dir: Unused.
      game_steps_per_episode: Unused.
      score_index: Unused.
      score_multiplier: Unused.
      random_seed: Unused.
      disable_fog: Unused.
      ensure_available_actions: Whether to throw an exception when an
        unavailable action is passed to step().
      version: Unused.
    Raises:
      ValueError: if args are passed.
    """
    del map_name  # Unused.
    del discount  # Unused.
    del discount_zero_after_timeout  # Unused.
    del visualize  # Unused.
    del step_mul  # Unused.
    del save_replay_episodes  # Unused.
    del replay_dir  # Unused.
    del game_steps_per_episode  # Unused.
    del score_index  # Unused.
    del score_multiplier  # Unused.
    del random_seed  # Unused.
    del disable_fog  # Unused.
    del ensure_available_actions  # Unused.
    del version  # Unused.

    if realtime:
      raise ValueError('realtime mode is not supported by the mock env.')

    if not players:
      players = [sc2_env.Agent(sc2_env.Race.random)]

    num_agents = sum(1 for p in players if isinstance(p, sc2_env.Agent))

    if agent_interface_format is None:
      raise ValueError('Please specify agent_interface_format.')

    if isinstance(agent_interface_format,
                  (sc2_env.AgentInterfaceFormat, sc2api_pb2.InterfaceOptions)):
      agent_interface_format = [agent_interface_format] * num_agents

    if len(agent_interface_format) != num_agents:
      raise ValueError(
          'The number of entries in agent_interface_format should '
          'correspond 1-1 with the number of agents.')

    self._game_info = _make_dummy_game_info(players, agent_interface_format)
    self._agent_interface_formats = agent_interface_format
    self._features = [
        features.features_from_game_info(
            game_info=g, agent_interface_format=aif)
        for g, aif in zip(self._game_info, self._agent_interface_formats)]

    super(SC2TestEnv, self).__init__(
        num_agents=num_agents,
        action_spec=tuple(f.action_spec() for f in self._features),
        observation_spec=tuple(f.observation_spec() for f in self._features))
    self.episode_length = 10

  @property
  def game_info(self):
    return self._game_info

  def save_replay(self, *args, **kwargs):
    """Does nothing."""

  def _default_observation(self, obs_spec, agent_index):
    """Returns a mock observation from an SC2Env."""

    builder = dummy_observation.Builder(obs_spec).game_loop(0)
    aif = self._agent_interface_formats[agent_index]
    if (isinstance(aif, sc2_env.AgentInterfaceFormat) and
        (aif.use_feature_units or aif.use_raw_units)):
      feature_units = [
          dummy_observation.FeatureUnit(
              units.Neutral.LabBot,
              features.PlayerRelative.NEUTRAL,
              owner=16,
              pos=common_pb2.Point(x=10, y=10, z=0),
              radius=1.0,
              health=5,
              health_max=5,
              is_on_screen=True,
          )
      ]

      builder.feature_units(feature_units)

    response_observation = builder.build()
    features_ = self._features[agent_index]
    observation = features_.transform_obs(response_observation)

    # Add bounding box for the minimap camera in top left of feature screen.
    if hasattr(observation, 'feature_minimap'):
      minimap_camera = observation.feature_minimap.camera
      minimap_camera.fill(0)
      height, width = [dim // 2 for dim in minimap_camera.shape]
      minimap_camera[:height, :width].fill(1)

    return observation


def _make_dummy_game_info(players, interface_formats):
  """Makes dummy game infos from player and interface format knowledge_data."""
  player_info = []
  for i, p in enumerate(players, start=1):
    if isinstance(p, sc2_env.Agent):
      player_info.append(
          sc2api_pb2.PlayerInfo(
              player_id=i,
              type=sc2api_pb2.PlayerType.Participant,
              race_requested=p.race[0],
              player_name=p.name))
    else:
      player_info.append(
          sc2api_pb2.PlayerInfo(
              player_id=i,
              type=sc2api_pb2.PlayerType.Computer,
              race_requested=p.race[0],
              difficulty=p.difficulty,
              ai_build=p.build[0],
              player_name=p.difficulty.name))

  game_infos = []
  for _, interface_format in zip(players, interface_formats):
    game_info = sc2api_pb2.ResponseGameInfo(
        player_info=player_info,
        start_raw=raw_pb2.StartRaw(
            map_size=common_pb2.Size2DI(x=DUMMY_MAP_SIZE, y=DUMMY_MAP_SIZE)))

    if isinstance(interface_format, sc2api_pb2.InterfaceOptions):
      game_info.options.CopyFrom(interface_format)
    else:
      if interface_format.feature_dimensions is not None:
        fd = interface_format.feature_dimensions
        game_info.options.feature_layer.resolution.x = fd.screen.x
        game_info.options.feature_layer.resolution.y = fd.screen.y
        game_info.options.feature_layer.minimap_resolution.x = fd.minimap.x
        game_info.options.feature_layer.minimap_resolution.y = fd.minimap.y
        game_info.options.feature_layer.width = (
            interface_format.camera_width_world_units)
      if interface_format.rgb_dimensions is not None:
        rd = interface_format.rgb_dimensions
        game_info.options.render.resolution.x = rd.screen.x
        game_info.options.render.resolution.y = rd.screen.y
        game_info.options.render.minimap_resolution.x = rd.minimap.x
        game_info.options.render.minimap_resolution.y = rd.minimap.y
        game_info.options.render.width = (
            interface_format.camera_width_world_units)

    game_infos.append(game_info)

  return game_infos

