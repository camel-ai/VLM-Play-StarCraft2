import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import math



"""
automatic annotate the unit on the image from pysc2 observation. 
pysc2 observation is a dictionary, we need to extract the unit information from the dictionary.

"""

def _annotate_units_on_image(image, units_to_annotate, circle_radius=25, font_scale=1, circle_thickness=2):
    """
    在图像上标注指定的单位，确保标注完整且清晰可读。

    Args:
        image: 原始图像
        units_to_annotate: 要标注的单位列表
        circle_radius: 圆圈半径 (最小值为15)
        font_scale: 字体大小 (最小值为0.6)
        circle_thickness: 圆圈线条粗细
    """
    if len(units_to_annotate) == 0:
        return image

    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    occupied_positions = []

    # 固定最小值以确保可读性
    MIN_CIRCLE_RADIUS = 15
    MIN_FONT_SCALE = 0.6

    # 计算图像边界安全区域
    safe_margin = circle_radius + 10  # 确保标注不会被切割

    # 计算有效标注区域
    valid_area = {
        'min_x': safe_margin,
        'max_x': image.shape[1] - safe_margin,
        'min_y': safe_margin,
        'max_y': image.shape[0] - safe_margin
    }

    # 计算图像中心
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    # 按距离排序，但给予边缘单位更高优先级
    def unit_priority(unit):
        pos = unit['position']
        # 检查是否接近边缘
        is_edge = (pos[0] < valid_area['min_x'] or
                   pos[0] > valid_area['max_x'] or
                   pos[1] < valid_area['min_y'] or
                   pos[1] > valid_area['max_y'])
        # 边缘单位优先，其次按距离中心远近排序
        return (not is_edge,
                (pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)

    sorted_units = sorted(units_to_annotate, key=unit_priority)

    for unit in sorted_units:
        number = unit['tag_index']
        color = unit['color']

        # 获取原始位置
        original_x = min(max(int(round(unit['position'][0])), 0), image.shape[1] - 1)
        original_y = min(max(int(round(unit['position'][1])), 0), image.shape[0] - 1)
        original_pos = (original_x, original_y)

        # 计算初始标注位置（考虑边界情况）
        initial_x = min(max(original_x, valid_area['min_x']), valid_area['max_x'])
        initial_y = min(max(original_y, valid_area['min_y']), valid_area['max_y'])

        # 基于初始位置寻找最优位置
        pos = find_optimal_position(
            target_pos=(initial_x, initial_y),
            original_pos=original_pos,
            occupied_positions=occupied_positions,
            circle_radius=max(circle_radius, MIN_CIRCLE_RADIUS),
            valid_area=valid_area,
            max_distance=circle_radius * 4
        )

        # 绘制圆圈
        cv2.circle(annotated_image, pos, max(circle_radius, MIN_CIRCLE_RADIUS),
                   color, circle_thickness)

        # 优化文字显示
        text = str(number)
        current_font_scale = max(font_scale, MIN_FONT_SCALE)

        # 计算并调整文字大小，确保在圆圈内
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, current_font_scale, circle_thickness
        )

        # 确保文字位置完全在圆圈内
        text_pos = (
            int(pos[0] - text_width / 2),
            int(pos[1] + text_height / 3)
        )

        # 绘制文字（带描边）
        cv2.putText(annotated_image, text, text_pos, font, current_font_scale,
                    (0, 0, 0), circle_thickness + 2, cv2.LINE_AA)
        cv2.putText(annotated_image, text, text_pos, font, current_font_scale,
                    color, circle_thickness, cv2.LINE_AA)

        occupied_positions.append(pos)

    return annotated_image


def find_optimal_position(target_pos, original_pos, occupied_positions, circle_radius,
                          valid_area, max_distance):
    """
    寻找最优标注位置，确保在有效区域内且尽可能靠近目标位置。
    """
    if not is_overlapping(target_pos, occupied_positions, circle_radius):
        return target_pos

    # 搜索方向优先级：靠近原始位置的方向
    dx = target_pos[0] - original_pos[0]
    dy = target_pos[1] - original_pos[1]

    # 基于原始位置与目标位置的关系确定搜索方向
    primary_directions = []
    if abs(dx) > abs(dy):
        primary_directions = [(1, 0), (-1, 0)] if dx > 0 else [(-1, 0), (1, 0)]
    else:
        primary_directions = [(0, 1), (0, -1)] if dy > 0 else [(0, -1), (0, 1)]

    # 添加对角线方向
    all_directions = primary_directions + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    best_pos = target_pos
    min_overlap_count = float('inf')
    min_distance = float('inf')

    # 在每个方向上搜索
    step = max(circle_radius // 3, 5)
    for direction in all_directions:
        for distance in range(step, max_distance + step, step):
            new_x = int(target_pos[0] + direction[0] * distance)
            new_y = int(target_pos[1] + direction[1] * distance)

            # 检查是否在有效区域内
            if (valid_area['min_x'] <= new_x <= valid_area['max_x'] and
                    valid_area['min_y'] <= new_y <= valid_area['max_y']):

                current_pos = (new_x, new_y)
                overlap_count = count_overlaps(current_pos, occupied_positions, circle_radius)
                dist = ((current_pos[0] - original_pos[0]) ** 2 +
                        (current_pos[1] - original_pos[1]) ** 2) ** 0.5

                if overlap_count < min_overlap_count or (
                        overlap_count == min_overlap_count and dist < min_distance
                ):
                    min_overlap_count = overlap_count
                    min_distance = dist
                    best_pos = current_pos

                if overlap_count == 0:
                    return best_pos

    return best_pos


def count_overlaps(pos, occupied_positions, circle_radius):
    """统计重叠数量"""
    return sum(1 for occ_pos in occupied_positions if
               is_overlapping(pos, [occ_pos], circle_radius))


def is_overlapping(pos, occupied_positions, circle_radius):
    """检查是否重叠"""
    min_distance = circle_radius * 2
    return any((pos[0] - occ[0]) ** 2 + (pos[1] - occ[1]) ** 2 < min_distance ** 2
               for occ in occupied_positions)

def draw_grid_with_labels(frame: np.ndarray, screen_size: tuple, grid_size: tuple,
                          grid_color=(200, 200, 200), grid_thickness=2,
                          grid_opacity=0.8, border_color=(150, 150, 150),
                          border_thickness=1) -> np.ndarray:
    """
    在图像上绘制网格和标签。

    Parameters:
    - frame: np.ndarray, 输入图像
    - screen_size: tuple(int, int), 屏幕的(宽度, 高度)
    - grid_size: tuple(int, int), 网格的(列数, 行数)
    - grid_color: tuple(int, int, int), 网格线的颜色，默认浅灰色
    - grid_thickness: int, 网格线的粗细
    - grid_opacity: float, 网格的不透明度 (0.0 到 1.0)
    - border_color: tuple(int, int, int), 单元格边框的颜色
    - border_thickness: int, 单元格边框的粗细

    Returns:
    - np.ndarray: 添加了网格和标签的图像
    """
    # 解包尺寸
    screen_width, screen_height = screen_size
    grid_columns, grid_rows = grid_size

    # 计算单元格尺寸
    cell_width = screen_width // grid_columns
    cell_height = screen_height // grid_rows

    # 创建网格覆盖层
    grid_overlay = frame.copy()

    # 绘制垂直网格线
    for i in range(1, grid_columns):
        x = i * cell_width
        cv2.line(grid_overlay, (x, 0), (x, screen_height), grid_color, grid_thickness)

    # 绘制水平网格线
    for i in range(1, grid_rows):
        y = i * cell_height
        cv2.line(grid_overlay, (0, y), (screen_width, y), grid_color, grid_thickness)

    # 混合网格覆盖层和原始帧
    cv2.addWeighted(grid_overlay, grid_opacity, frame, 1 - grid_opacity, 0, frame)

    # 绘制单元格边框
    for i in range(grid_columns):
        for j in range(grid_rows):
            x1, y1 = i * cell_width, j * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

    # 添加标签
    # 根据图像尺寸动态计算字体大小
    base_font_scale = min(screen_width, screen_height) / 1920  # 以1920px为基准
    font_scale = max(0.6, min(1.5, base_font_scale))  # 限制范围

    # 设置字体属性
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = max(2, int(3 * base_font_scale))
    font_color = (255, 255, 255)  # 白色

    # 计算文本大小以优化放置位置
    test_text = "0"
    (text_w, text_h), baseline = cv2.getTextSize(test_text, font, font_scale, font_thickness)

    # 计算边距
    margin_top = int(screen_height * 0.03)  # 顶部边距
    margin_left = int(screen_width * 0.02)  # 左侧边距

    # 绘制横向(x轴)标签
    for i in range(grid_columns):
        x = i * cell_width + cell_width // 2
        label_y = margin_top + text_h

        # 绘制带描边的文字
        cv2.putText(frame, str(i), (x - text_w // 2, label_y), font, font_scale,
                    (0, 0, 0), font_thickness + 2)  # 黑色描边
        cv2.putText(frame, str(i), (x - text_w // 2, label_y), font, font_scale,
                    font_color, font_thickness)  # 白色文字

    # 绘制纵向(y轴)标签
    for i in range(grid_rows):
        y = i * cell_height + cell_height // 2

        # 绘制带描边的文字
        cv2.putText(frame, str(i), (margin_left, y + text_h // 2), font, font_scale,
                    (0, 0, 0), font_thickness + 2)  # 黑色描边
        cv2.putText(frame, str(i), (margin_left, y + text_h // 2), font, font_scale,
                    font_color, font_thickness)  # 白色文字

    return frame
