import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
ROOT_DIR = os.path.dirname(current_dir)

# 定义position文件的相对路径
POSITION_FILE_RELATIVE_PATH = os.path.join('vlm_attention', 'env', 'labeld_unit_positions.json')
CONFIG_FILE_RELATIVE_PATH = os.path.join('vlm_attention', 'env', 'config.json')

