import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def annotate_units_on_image(image, units_to_annotate, circle_radius=25, font_scale=1, circle_thickness=2):
    """
    在图像上标注指定的单位，避免重叠，并添加详细的日志。

    :param image: 原始图像
    :param units_to_annotate: 要标注的单位列表，每个单位是一个字典，包含 'tag_index', 'position', 'color'
    :param circle_radius: 圆圈半径
    :param font_scale: 字体大小
    :param circle_thickness: 圆圈线条粗细
    :return: 标注后的图像
    """
    # logger.info(f"开始标注图像。图像大小: {image.shape}, 要标注的单位数量: {len(units_to_annotate)}")

    if len(units_to_annotate) == 0:
        # logger.warning("没有要标注的单位！")
        return image

    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    occupied_positions = []

    for i, unit in enumerate(units_to_annotate):
        number = unit['tag_index']
        original_pos = unit['position']
        color = unit['color']

        # logger.debug(
        #     f"正在处理第 {i + 1}/{len(units_to_annotate)} 个单位。标签: {number}, 位置: {original_pos}, 颜色: {color}")

        # 检查位置是否在图像范围内
        if not (0 <= original_pos[0] < image.shape[1] and 0 <= original_pos[1] < image.shape[0]):
            # logger.warning(f"单位 {number} 的位置 {original_pos} 超出图像范围！跳过此单位。")
            continue

        # 找到一个不重叠的位置
        pos = find_non_overlapping_position(original_pos, occupied_positions, circle_radius)
        # logger.debug(f"找到的非重叠位置: {pos}")

        # 绘制圆圈
        cv2.circle(annotated_image, pos, circle_radius, color, circle_thickness)
        # logger.debug(f"已绘制圆圈，中心: {pos}, 半径: {circle_radius}, 颜色: {color}")

        # 绘制数字
        text = str(number)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, circle_thickness)
        text_pos = (
            int(pos[0] - text_width / 2),
            int(pos[1] + text_height / 2)
        )
        cv2.putText(annotated_image, text, text_pos, font, font_scale, color, circle_thickness, cv2.LINE_AA)
        # logger.debug(f"已绘制文本 '{text}'，位置: {text_pos}")

        # 将这个位置添加到已占用列表中
        occupied_positions.append(pos)

    # logger.info(f"图像标注完成。共标注了 {len(occupied_positions)} 个单位。")

    # 检查标注后的图像是否与原图有差异
    # if np.array_equal(image, annotated_image):
    #     logger.warning("警告：标注后的图像与原图相同，可能没有成功添加标注！")
    # else:
    #     logger.info("标注后的图像与原图有差异，标注可能已成功添加。")

    return annotated_image

def find_non_overlapping_position(original_pos, occupied_positions, circle_radius):
    """
    找到一个不与已有标注重叠的位置。
    """
    pos = original_pos
    offset = circle_radius * 2
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上、右、下、左
    current_direction = 0
    steps = 1

    while is_overlapping(pos, occupied_positions, circle_radius):
        dx, dy = directions[current_direction]
        pos = (int(pos[0] + dx * offset), int(pos[1] + dy * offset))

        steps -= 1
        if steps == 0:
            current_direction = (current_direction + 1) % 4
            if current_direction % 2 == 0:
                steps += 1

    return pos

def is_overlapping(pos, occupied_positions, circle_radius):
    """
    检查给定位置是否与已有的标注重叠。
    """
    for occupied_pos in occupied_positions:
        distance = np.sqrt((pos[0] - occupied_pos[0])**2 + (pos[1] - occupied_pos[1])**2)
        if distance < circle_radius * 2:
            return True
    return False

def _annotate_units_on_image(image, units_to_annotate):
    """
    在图像上标注指定的单位。

    :param image: 原始图像
    :param units_to_annotate: 要标注的单位列表，每个单位是一个字典，包含 'tag_index', 'position', 'color'
    :return: 标注后的图像
    """
    return annotate_units_on_image(image, units_to_annotate)

def _draw_grid (frame, screen_size, grid_size):
    """
    Draw a grid on the given frame with customizable parameters.

    Parameters:
    - frame: np.ndarray, the input image on which to draw the grid
    - screen_size: tuple(int, int), the (width, height) of the frame
    - grid_size: tuple(int, int), the number of (columns, rows) in the grid

    Returns:
    - np.ndarray, the frame with the grid drawn on it

    frame: 输入的图像帧，我们将在其上绘制网格。
    screen_size: 包含屏幕宽度和高度的元组。
    grid_size: 包含网格列数和行数的元组。
    cell_width 和 cell_height: 根据屏幕尺寸和网格大小计算的单元格宽度和高度。
    grid_overlay: 原始帧的副本，用于绘制网格线。
    grid_color: 网格线的颜色，默认为浅灰色。
    grid_thickness: 网格线的粗细。
    grid_opacity: 网格的不透明度，范围从 0.0（完全透明）到 1.0（完全不透明）。
    border_color: 单元格边框的颜色，默认为深灰色。
    border_thickness: 单元格边框的粗细。

    要调整网格的可见度：

    增加 grid_opacity 的值会使网格更加明显。
    调整 grid_color 可以改变网格线的颜色。
    增加 grid_thickness 会使网格线更粗。
    调整 border_color 和 border_thickness 可以改变单元格边框的外观。
    """
    # Unpack dimensions
    screen_width, screen_height = screen_size
    grid_columns, grid_rows = grid_size

    # Calculate cell dimensions
    cell_width = screen_width // grid_columns
    cell_height = screen_height // grid_rows

    # Create a copy of the frame for the grid overlay
    grid_overlay = frame.copy()

    # Grid line parameters
    grid_color = (200, 200, 200)  # Light gray color for grid lines
    grid_thickness = 2  # Thickness of grid lines

    # Draw vertical grid lines
    for i in range(1, grid_columns):
        x = i * cell_width
        cv2.line(grid_overlay, (x, 0), (x, screen_height), grid_color, grid_thickness)

    # Draw horizontal grid lines
    for i in range(1, grid_rows):
        y = i * cell_height
        cv2.line(grid_overlay, (0, y), (screen_width, y), grid_color, grid_thickness)

    # Blend the grid overlay with the original frame
    grid_opacity = 0.8  # Opacity of the grid (0.0 to 1.0)
    cv2.addWeighted(grid_overlay, grid_opacity, frame, 1 - grid_opacity, 0, frame)

    # Cell border parameters
    border_color = (150, 150, 150)  # Darker gray for cell borders
    border_thickness = 1  # Thickness of cell borders

    # Draw borders for each cell
    for i in range(grid_columns):
        for j in range(grid_rows):
            x1, y1 = i * cell_width, j * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

    return frame