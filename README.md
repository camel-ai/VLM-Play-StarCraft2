# StarCraft II Multimodal Agent Project

## 项目概述

这个项目旨在开发一个用于星际争霸II（StarCraft II）的多模态智能体（Multimodal Agent）。该智能体能够处理游戏的视觉信息和文本描述，并做出相应的决策。

## 主要特性

- 基于 OpenAI Gym 的自定义环境
- 支持多模态输入（图像和文本）
- 自动单位标注系统
- 可配置的单位标识和颜色方案
- 随机行为智能体用于测试

## 环境设置

### 依赖项

- Python 3.10
- 操作系统: Windows 11(linux没有星际的图形界面，但我们现在可以直接读取rgb值现在版本的代码,linux理论上也应该有图形), linux和mac没有试过(我的服务器有点问题，希望大家有服务器在服务器测试一下)
- requirement: `vlm_attention/requirements.txt`

### 安装

1. 克隆仓库：
2. 下载星际争霸2, 请登录亚服战网下载. Linux 请在https://github.com/Blizzard/s2client-proto#downloads 下载4.10.0版本
3. 地图设置.在星际争霸2 游戏文件夹下的Maps文件夹(如果没有请创建一个),然后将`pysc2/maps/SMAC`整个文件夹放入Maps文件夹下,我们此次实验的地图为`vlm_attention_1.SC2Map`
4. 如果你想使用自定义地图，你需要在查看该文件`pysc2/maps/readme.md`


## 地图
1. 单个玩家且没有技能释放的(single player without ability)
   `[2bc1prism_vs_8m_vlm_attention.SC2Map]`
3. 两个玩家且没有技能的(2 player without ability)
4. 单个玩家且有技能的(single player ability)
5. 两个玩家且有技能的(2 player ability)
## 配置

项目使用 `vlm_attention/env/config.py` 文件进行配置。主要配置项包括：
```python
# 颜色定义 (BGR 格式)
COLORS = {
    "self_color": [0, 255, 0],  # 绿色
    "enemy_color": [255, 255, 0]  # 黄色
}

# 从pysc2 读取标准单位名称
 def get_unit_name(unit_type: int) -> str:
    """
    通过unit_type获取单位名称
    使用pysc2.lib.units中的枚举定义

    Args:
        unit_type: 单位类型ID

    Returns:
        str: 格式化的单位名称，如果未找到则返回Unknown
    """
    # 按种族顺序尝试获取单位类型
    for race in [Protoss, Terran, Zerg, Neutral]:
        try:
            unit_enum = race(unit_type)
            return format_unit_name(unit_enum.name)
        except ValueError:
            continue

    return f"Unknown Unit {unit_type}"
```

## 使用

1. 每次安装时请注意运行vlm_attention/knowledge_data/database/generate_config_yaml.py,以确保数据库连接正确
````python
def main():
    folder_path = r"处理后的json文件夹路径" # 这里是需要修改的
    output_file = "sc2_unit_data_index.yaml"

.......
````
3. 运行主程序,有单进程和多进程：
```bash
python run_env.py # 单进程
python multiprocess_run_env.py # 多进程

# 注意,目前请使用`vlm_agent_without_move_v5`,`test_agent`进行测试与修改.
```

这里我们默认配置文件已提供相关参数,你可以直接运行,也可以根据自己的需要进行修改
例如:
```bash
python run_env.py --map vlm_attention_1
```


## 项目结构

1. `env_bot.py`: 定义了与星际争霸II交互的 Multimodal_bot 类
2. `env_core.py`: 实现了自定义的 SC2MultimodalEnv 环境
3. `run_env.py`: 主运行脚本，包含 RandomAgent与TestAgent 用于测试
4. `utils.py`: 包含一些辅助函数,包括调用llm,vlm和测试api, 注意由于我在大陆,所以用了proxy确保运行.自己部署时请按照需要修改`vlm_attention/utils/call_vlm.py`
```python

class BaseChatbot:
    def __init__(self, model_name):
        # 设置代理
        proxy_url = get_config("proxy", "url")
        os.environ["http_proxy"] = proxy_url # 设置代理
        os.environ["https_proxy"] = proxy_url # 设置代理
        

```
6. `knowledge_data`: 包含了一些知识数据,其中`\database` 是数据库和对应的调用代码,而`\firecrawl_test`是爬取数据和清洗数据的代码

## 自定义和拓展
1. 修改 `config.json` 以适应不同的游戏场景和单位配置
2. 在 `env_bot.py` 中扩展 Multimodal_bot 类以实现更复杂的行为. 目前只支持进攻行为,比如3号单位进攻14号单位
3. 在 `run_env.py` 中替换 FlexibleRandomAgent和TestAgent 以实现更高级的决策逻辑
## 环境设置

### 动作空间

- `attack`: 进攻指定单位. 
- `move`: 移动到指定位置

### 观测空间

观测空间是一个字典，包含三个关键元素：
```python
{
            'text': text_description,
            'image': raw_image,
            'unit_info': unit_info
        }
```
        
1. **文本观测** (`'text'`)
- 描述：提供游戏状态的文本描述，包括我方和敌方单位的信息。只有第一个step才会有文本观测.

2. **图像观测** (`'image'`)
- 描述：游戏画面的截图。每个step都会有图像观测,这是从Obs中获取的

3. **单位信息** (`'unit_info'`)
- 描述：包含场上单位的详细信息，用于标注和决策。利用feature screen获取.

- 具体信息如下
```python
{
                            'tag_index': tag_index, # 标签
                            'position': (screen_x, screen_y), # 位置
                            'color': color, # 颜色
                            'alliance': unit.alliance # 阵营
                        }
```



### 动作空间
`_generate_random_actions`
方法返回一个字典，包含两个键：attack 和 move，每个键对应一个包含一个或多个元组的列表。

- 攻击动作 (attack)
  - 每个元组包含两个整数 (攻击者标签, 目标标签)。 例如：[(1, 6)] 表示标签为1的我方单位攻击标签为6的敌方单位。
          空列表 [] 表示不执行任何攻击动作。 
  - 例如：[(1, 6)] 表示标签为1的我方单位攻击标签为6的敌方单位。 
  - 空列表 [] 表示不执行任何攻击动作。


- 移动动作 (move)
  - move_type:
    - 0: 不移动 
    - 1: 原始移动 (使用网格坐标)  ； 默认10 x 10
    - 2: SMAC移动 (使用方向，0,1,2,3对应上下左右)
  - unit_index: 要移动的单位索引
  - target:
    - 对于原始移动 (move_type=1): [x, y] 网格坐标 
    - 对于SMAC移动 (move_type=2): [direction]，方向定义如下： 
      - 0: 向上移动 
      - 1: 向右移动 
      - 2: 向下移动 
      - 3: 向左移动 
    - 对于不移动 (move_type=0): [0, 0]




### 选择单位标注
`annotate_units_on_image` 需要一个字典作为输入，包含单位标签和颜色信息。返回一个标注过的图像，用于显示单位信息。

例如
```python
units_to_annotate = [
    {'tag_index': 1, 'position': (100, 100), 'color': (0, 255, 0)},  # 绿色标注
    {'tag_index': 2, 'position': (200, 200), 'color': (0, 0, 255)},  # 红色标注
    {'tag_index': 3, 'position': (300, 300), 'color': (255, 0, 0)}   # 蓝色标注
]
```

### Agent 设置

目前由于移动和网格功能反馈不佳,我们提供了一个只调用攻击不移动的agent.

此外agent的一些配置信息和辅助函数在`\run_env\agent\agent_utils.py`和`run_env\utils.py`

具体需要设置的参数为:
1. use_self_attention: 是否使用自注意力
2. use_rag: 是否使用RAG

```python

class VLMAgentWithoutMove:
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, grid_size: Tuple[int, int] = (10, 10),
                 use_self_attention: bool = False, use_rag: bool = False):
        
        """
        初始化VLMAgentWithoutMove代理。
        :param action_space: 动作空间字典
        :param config_path: 配置文件路径
        :param save_dir: 保存目录
        :param draw_grid: 是否在截图上绘制网格
        :param annotate_units: 是否在截图上标注单位
        :param grid_size: 网格大小
        :param use_self_attention: 是否使用自注意力
        :param use_rag: 是否使用RAG
        """

```

### 星际争霸2 observation信息

我们在`vlm_attention/env/multiprocess_test/test.py`进行测试,对观测空间进行处理,得到对应的reward和score信息.

```python
# Reward & score

            step_count += 1
            total_reward += obs[0].reward

            if obs[0].last():
                score = obs[0].observation["score_cumulative"]
                print(f"Episode {episode + 1} Results:")
                print(f"  Steps: {step_count}")
                print(f"  Total Reward: {total_reward}")
                print(f"  Total Score: {score[0]}")
                print(f"  Collected Minerals: {score[1]}")
                print(f"  Collected Vespene: {score[2]}")
                print(f"  Collected Resources: {score[3]}")
                print(f"  Spent Minerals: {score[4]}")
                print(f"  Spent Vespene: {score[5]}")
                print(f"  Food Used (Supply): {score[6]}")
                print(f"  Killed Unit Score: {score[7]}")
                print(f"  Killed Building Score: {score[8]}")
                print(f"  Killed Minerals: {score[9]}")
                print(f"  Killed Vespene: {score[10]}")
                print(f"  Lost Minerals: {score[11]}")
                print(f"  Lost Vespene: {score[12]}")
                print(f"  Friendly Fire Minerals: {score[13]}")
                print(f"  Friendly Fire Vespene: {score[14]}")
                print(f"  Used Minerals: {score[15]}")
                print(f"  Used Vespene: {score[16]}")
                print(f"  Total Used Resources: {score[17]}")
                print(f"  Total Items Score: {score[18]}")
                print(f"  Score: {score[19]}")

```

完整的观测结构保存在`vlm_attention/env/multiprocess_test/obs_structure.json`


### RAG
1. 目前使用firecrawl先爬取网站`https://liquipedia.net/starcraft2/Main_Page`的数据,进行清洗和处理.爬取后的数据存放在`vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info`中.
清晰后的数据存放在`vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info_processed`.清洗代码为`vlm_attention\knowledge_data\firecrawl_test\clean_sc2_unit_info_file.py`.
2. 请注意,我们是预先爬取然后清洗的,不涉及运行时新增的情况.同时爬取所需的url存放在`vlm_attention\knowledge_data\url.py`中,如果需要爬取新的数据请自行添加
3. 首次安装时请运行`vlm_attention\knowledge_data\database\generate_config_yaml.py`文件,注意需要替换对应的`folder_path`为你本地电脑下`vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info`的绝对位置
4. 安装好后请运行`vlm_attention/knowledge_data/database/test_llm_use_database.py`进行测试.

### 2024-10-25

目前已实现从游戏引擎获取rgb图像,从游戏引擎获取unit.tag,支持多个同种单位操作(比如3个stalker). 

### 2024-11-05
更新新地图,请从pysc2/maps/VLM_ATTENTION 将地图放置在SC2/MAPS文件夹下面. 新地图为2cvs64zg 的vlm_attention版本.
```python
目前可选地图为:
map_list = ["vlm_attention_1",
            "2c_vs_64zg_vlm_attention",
            "2m_vs_1z_vlm_attention",
            "2s_vs_1sc_vlm_attention",
            "2s3z_vlm_attention",
            "3m_vlm_attention",
            "3s_vs_3z_vlm_attention"]
```

更新了调用模型的vlm_attention/utils/call_vlm.py. 现在默认为


```python
class TextChatbot(BaseChatbot):
    def query(self, system_prompt, user_input, maintain_history=True):

class MultimodalChatbot(BaseChatbot):
    def query(self, system_prompt, user_input, image_path=None, maintain_history=True):

```
