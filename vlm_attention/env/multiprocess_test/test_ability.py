from absl import app, flags
import logging
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features

FLAGS = flags.FLAGS

class SimpleAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SimpleAgent, self).__init__()
        self.abilities_checked = False
        self.unit_selected = False

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        # 1. 搜索屏幕上的单位并选择
        if not self.unit_selected:
            try:
                # 查找属于玩家的单位
                units_on_screen = [unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.SELF]
                if units_on_screen:
                    # 选择第一个单位
                    selected_unit = units_on_screen[0]
                    target = (selected_unit.x, selected_unit.y)
                    print(f"Selected Unit: ID={selected_unit.unit_type}, Position=({target[0]}, {target[1]})")
                    self.unit_selected = True
                    return actions.FUNCTIONS.select_point("select", target)
                else:
                    print("No units found on screen.")
                    return actions.FUNCTIONS.no_op()

            except Exception as e:
                logging.error(f"Error: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())

        # 2. 检查选中单位的可用能力
        elif self.unit_selected and not self.abilities_checked:
            print("\n=== Available Abilities for Selected Unit ===")
            available_abilities = obs.observation.available_actions
            if len(available_abilities) > 0:  # 显式检查是否有可用能力
                for action_id in available_abilities:
                    action = actions.FUNCTIONS[action_id]
                    print(f"Action ID: {action_id}, Name: {action.name}")
            else:
                print("No abilities available for the selected unit.")
            self.abilities_checked = False

        return actions.FUNCTIONS.no_op()


def main(argv):
    agent = SimpleAgent()

    try:
        with sc2_env.SC2Env(
                map_name="MMM_vlm_attention_two_players",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True,
                    use_unit_counts=True,
                    use_camera_position=True,
                    action_space=actions.ActionSpace.FEATURES,
                ),
                step_mul=8,
                game_steps_per_episode=1000,  # Maximum game steps per episode
                visualize=True
        ) as env:

            agent.setup(env.observation_spec(), env.action_spec())

            timesteps = env.reset()
            agent.reset()

            # Run until the abilities are checked
            while not agent.abilities_checked:
                step_actions = [agent.step(timesteps[0])]

                # If the game ends before abilities are checked
                if timesteps[0].last():
                    break

                timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
