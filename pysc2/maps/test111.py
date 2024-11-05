from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import point

import pygame

# 创建 StarCraft II 环境
env = sc2_env.SC2Env(
    map_name="3m",
    players=[sc2_env.Agent(sc2_env.Race.terran)],
    agent_interface_format=features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=84, minimap=64),
        use_feature_units=True,
    ),
    step_mul=8,
    game_steps_per_episode=0,
    visualize=True,
)

# 创建 Pygame 窗口
pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

# 游戏循环
while True:
    # 重置环境
    obs = env.reset()[0]

    while True:
        # 渲染游戏画面
        screen.fill((0, 0, 0))
        minimap = obs.feature_minimap
        minimap_surf = pygame.surfarray.make_surface(minimap.transpose((1, 0, 2)))
        screen.blit(minimap_surf, (0, 0))

        # 获取我方智能体的位置和编号
        for unit in obs.observation.feature_units:
            if unit.alliance == features.PlayerRelative.SELF:
                pos = point.Point(unit.x, unit.y)
                tag = unit.tag
                # 在智能体位置绘制编号
                text = str(tag)
                font = pygame.font.Font(None, 24)
                text_surface = font.render(text, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(pos.x, pos.y))
                screen.blit(text_surface, text_rect)

        pygame.display.flip()
        clock.tick(60)

        # 获取用户输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        # 环境步进
        actions = [sc2_env.Action()]
        obs = env.step(actions)[0]

        # 检查游戏是否结束
        if obs.last():
            break

# 关闭环境
env.close()