# Pysc2 地图载入机制
地图的载入实际上分两部分
1. 在pysc2-master下的pysc2/maps文件夹下,我们需要根据 `pysc2/maps/melee.py`和`pysc2/maps/mini_games.py`来编写我们对应地图的载入设置.然后在`pysc2/maps/__init__.py`导入我们对应的py文件
2. 地图需要存放在游戏客户端下的Maps文件夹下,如果没有Maps文件夹,请自行创建.然后将我们的地图文件夹放入Maps文件夹下即可.名称记得和````
3. 下方设计类的时候，directory的值需要和Maps文件夹下的地图文件夹名称一致，否则会报错  
```
class Melee(lib.Map):
  directory = "Melee"
  download = "https://github.com/Blizzard/s2client-proto#map-packs"
  players = 2
  game_steps_per_episode = 16 * 60 * 30  # 30 minute limit.
