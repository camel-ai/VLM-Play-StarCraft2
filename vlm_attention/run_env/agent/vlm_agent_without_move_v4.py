import json
import logging
import os
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import datetime
from vlm_attention.config.config import get_config
from vlm_attention.knowledge_data.database.sc2_unit_database import SC2UnitDatabase
from vlm_attention.run_env.agent.agent_utils import (
    summarize_unit_info, generate_important_units_prompt,
    generate_decision_prompt, format_units_info, parse_vlm_response, parse_vlm_decision,
    format_history_for_prompt, generate_enhanced_unit_selection_prompt,
    generate_unit_info_summary_prompt, generate_action_normalization_prompt
)
from vlm_attention.run_env.utils import _annotate_units_on_image, _draw_grid
from vlm_attention.utils.call_vlm import MultimodalChatbot, TextChatbot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLMAgentWithoutMove:
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, annotate_all_units: bool = True, grid_size: Tuple[int, int] = (10, 10)):
        self.action_space = action_space
        self.vlm = MultimodalChatbot(model_name=get_config("openai", "vlm_model_name"))
        self.text_bot = TextChatbot(model_name=get_config("openai", "llm_model_name"))
        self.important_units: List[int] = []
        self.text_observation: str = ""
        self.save_original_images = True
        self.history_length = 3
        self.history: List[Dict[str, Any]] = []
        self.draw_grid = draw_grid
        self.annotate_units = annotate_units
        self.annotate_all_units = annotate_all_units
        self.grid_size = grid_size
        self.step_count = 0

        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.unit_align_dict = self.config['unit_align_dict']
        self.predefined_tags = self.config['predefined_tags']

        self.friendly_units = [int(index) for tag, index in self.predefined_tags.items() if tag.endswith('_PLAYER_SELF')]
        self.enemy_units = [int(index) for tag, index in self.predefined_tags.items() if tag.endswith('_PLAYER_ENEMY')]

        self.save_dir = save_dir
        self.original_images_dir = os.path.join(self.save_dir, "original_images")
        self.first_annotation_dir = os.path.join(self.save_dir, "first_annotation")
        self.second_annotation_dir = os.path.join(self.save_dir, "second_annotation")
        self.vlm_io_dir = os.path.join(self.save_dir, "vlm_io")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        yaml_path = os.path.join(project_root, 'vlm_attention', 'knowledge_data', 'database', 'sc2_unit_data_index.yaml')

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found at {yaml_path}")

        self.db = SC2UnitDatabase(yaml_path)

        for directory in [self.save_dir, self.original_images_dir, self.first_annotation_dir,
                          self.second_annotation_dir, self.vlm_io_dir]:
            os.makedirs(directory, exist_ok=True)

        logger.info(f"VLMAgent initialized. Data will be saved in: {self.save_dir}")
        logger.info(f"Using YAML file: {yaml_path}")

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, List[Tuple]]:
        # 步骤 1: 初始化和日志记录
        self.step_count += 1
        self.text_observation = observation["text"]
        logger.info(f"Processing step {self.step_count}")
        logger.info(f"Text observation: {self.text_observation}")

        # 步骤 2: 保存原始图像（如果启用）
        if self.save_original_images:
            self._save_image(observation['image'], self.original_images_dir, "original")

        # 步骤 3: 第一次图像处理和重要单位识别
        first_annotation_path = self._process_and_save_image(observation, self.first_annotation_dir, "first_annotation",
                                                             annotate_all=True)
        important_units_response = self._identify_important_units(observation, first_annotation_path)
        self.important_units = parse_vlm_response(important_units_response)
        logger.info(f"Identified important units: {self.important_units}")

        # 步骤 4: 单位选择和信息查询
        available_units = self.db.get_all_unit_names()
        unit_selection_prompt = generate_enhanced_unit_selection_prompt(available_units, important_units_response,
                                                                        self.text_observation)
        units_to_query = self.text_bot.query(system_prompt="You are a StarCraft 2 strategist.",
                                             user_input=unit_selection_prompt)
        self._save_vlm_io(unit_selection_prompt, units_to_query, "unit_selection")
        logger.info(f"选择查询的单位：{units_to_query}")

        # 步骤 5: 获取并总结单位信息
        unit_list = [unit.strip() for unit in units_to_query.split(',')]
        unit_info = self._get_unit_info_from_database(unit_list)
        unit_summary_prompt = generate_unit_info_summary_prompt(unit_info)
        unit_summary = self.text_bot.query(
            system_prompt="You are a StarCraft 2 expert focusing on Protoss micro-management.",
            user_input=unit_summary_prompt)
        self._save_vlm_io(unit_summary_prompt, unit_summary, "unit_summary")
        logger.info(f"单位信息总结：\n{unit_summary}")

        # 步骤 6: 第二次图像处理
        second_annotation_path = self._process_and_save_image(observation, self.second_annotation_dir,
                                                              "second_annotation", annotate_all=False)

        # 步骤 7: 决策生成
        decision_system_prompt = generate_decision_prompt()
        decision_prompt = self._generate_decision_prompt(observation, unit_summary, important_units_response)
        raw_decision_response = self.vlm.query(system_prompt=decision_system_prompt,
                                               user_input=decision_prompt,
                                               image_path=second_annotation_path,
                                               maintain_history=True)
        self._save_vlm_io(decision_prompt, raw_decision_response, "raw_decision", image_path=second_annotation_path)

        # 步骤 8: 决策解析和规范化
        raw_action = parse_vlm_decision(raw_decision_response)
        normalization_prompt = generate_action_normalization_prompt(self.text_observation,
                                                                    format_units_info(observation.get('unit_info', []),
                                                                                      self.predefined_tags,
                                                                                      self.unit_align_dict),
                                                                    raw_action)

        max_retries = 3
        for attempt in range(max_retries):
            normalized_action_response = self.text_bot.query(
                system_prompt="""You are a StarCraft 2 expert tasked with reviewing and normalizing actions. 
                Your output must strictly follow this format:
                ## Actions ##
                Attack: [Attacker Tag] -> [Target Tag]
                Reasoning: [Brief explanation]

                Repeat this format for each attack action. Ensure all attacker tags are our units and all target tags are enemy units.

                Example output:
                ## Actions ##
                Attack: 1 -> 9
                Reasoning: The Stalker focuses on the Ghost due to its high damage potential and disabling abilities.

                Attack: 2 -> 14
                Reasoning: The Phoenix targets the weakened Banshee to eliminate its air-to-ground threat efficiently.

                Attack: 5 -> 12
                Reasoning: The Immortal targets the Medivac to prevent enemy healing and weaken overall sustain.""",
                user_input=normalization_prompt
            )
            self._save_vlm_io(normalization_prompt, normalized_action_response,
                              f"normalized_decision_attempt_{attempt + 1}")

            normalized_action = parse_vlm_decision(normalized_action_response)
            if normalized_action['attack']:
                break
            elif attempt < max_retries - 1:
                logger.warning(f"Failed to parse actions on attempt {attempt + 1}. Retrying...")
            else:
                logger.error("Failed to parse actions after maximum retries. Returning empty actions.")

        # 步骤 9: 添加无移动动作
        no_move_actions = [(0, unit, [0, 0]) for unit in self.friendly_units]
        normalized_action['move'] = no_move_actions

        # 步骤 10: 更新历史记录
        self._update_history(self.important_units, {'attack': normalized_action['attack']})

        # 步骤 11: 记录最终输出给环境的决策
        final_decision = {
            'attack': normalized_action['attack'],
            'move': normalized_action['move']
        }
        self._save_vlm_io("Final decision", json.dumps(final_decision), "final_decision")

        # 步骤 12: 返回最终动作
        return final_decision

    def _identify_important_units(self, observation: Dict[str, Any], image_path: str) -> str:
        system_prompt = generate_important_units_prompt()
        user_input = f"""
        Analyze the current StarCraft II game state based on the following information:

        Screenshot observation:
        {observation.get("text", "No text observation available.")}

        Units information:
        {format_units_info(observation.get('unit_info', []), self.predefined_tags, self.unit_align_dict)}

        Previous steps information:
        {format_history_for_prompt(self.history)}

        Based on this information, identify and explain the most strategically important enemy units.
        """

        important_units_response = self.vlm.query(system_prompt=system_prompt,
                                                  user_input=user_input,
                                                  image_path=image_path,
                                                  maintain_history=False)
        self._save_vlm_io(user_input, important_units_response, "important_units")

        return important_units_response

    def _generate_decision_prompt(self, observation: Dict[str, Any], unit_summary: str, important_units_response: str) -> str:
        return f"""
        Analyze the current StarCraft II game state based on the following information and suggest the best actions for our units:

        Screenshot observation:
        {self.text_observation}

        Units information:
        {format_units_info(observation.get('unit_info', []), self.predefined_tags, self.unit_align_dict)}

        Important enemy units analysis:
        {important_units_response}

        Unit information summary:
        {unit_summary}

        Previous steps information:
        {format_history_for_prompt(self.history)}

        Based on this information and the micro-management principles, provide attack actions for each of our units.
        """

    def _process_and_save_image(self, observation: Dict[str, Any], directory: str, prefix: str, annotate_all: bool) -> str:
        frame = observation['image']
        unit_info = observation['unit_info']
        logger.debug("Processing unit_info for image annotation")

        if self.draw_grid:
            logger.debug("Drawing grid on frame")
            frame = self._draw_grid_with_labels(frame)

        if self.annotate_units:
            if annotate_all:
                units_to_annotate = [unit['tag_index'] for unit in unit_info]
            else:
                units_to_annotate = self.friendly_units + self.important_units
            frame = self.annotate_units_on_frame(frame, unit_info, units_to_annotate)

        filename = os.path.join(directory, f"{prefix}_{self.step_count:04d}.png")
        cv2.imwrite(filename, frame)
        logger.debug(f"{prefix.capitalize()} saved: {filename}")
        return filename

    def _draw_grid_with_labels(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        screen_size = (w, h)

        frame_with_grid = _draw_grid(frame, screen_size, self.grid_size)

        cell_w, cell_h = w // self.grid_size[0], h // self.grid_size[1]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        font_color = (255, 255, 255)  # White color

        for i in range(self.grid_size[0]):
            x = i * cell_w + cell_w // 2
            cv2.putText(frame_with_grid, str(i), (x - 10, 30), font, font_scale, (0, 0, 0), font_thickness + 1)
            cv2.putText(frame_with_grid, str(i), (x - 10, 30), font, font_scale, font_color, font_thickness)

        for i in range(self.grid_size[1]):
            y = i * cell_h + cell_h // 2
            cv2.putText(frame_with_grid, str(i), (10, y + 10), font, font_scale, (0, 0, 0), font_thickness + 1)
            cv2.putText(frame_with_grid, str(i), (10, y + 10), font, font_scale, font_color, font_thickness)

        return frame_with_grid

    def _get_unit_info_from_database(self, unit_names: List[str]) -> Dict[str, Dict]:
        unit_info = {}
        for unit in unit_names:
            info = self.db.get_unit_info(unit)
            if info:
                unit_info[unit] = info
            else:
                logger.warning(f"Can't find unit information: {unit}")
        return unit_info

    def _summarize_unit_info(self, unit_info: Dict[str, Dict]) -> str:
        summary_prompt = summarize_unit_info(unit_info)
        return self.text_bot.query(
            "You are a StarCraft 2 expert. Summarize the following unit information concisely.", summary_prompt)

    def _save_vlm_io(self, prompt: str, response: str, prefix: str, image_path: Optional[str] = None) -> None:
        filename = os.path.join(self.vlm_io_dir, f"{prefix}_io_{self.step_count:04d}.json")
        data = {
            "prompt": prompt,
            "response": response,
            "step": self.step_count,
            "timestamp": datetime.datetime.now().isoformat()
        }

        if image_path:
            data["image_path"] = image_path

        try:
            if prefix in ["raw_decision", "normalized_decision"]:
                parsed_result = parse_vlm_decision(response)
                data["parsed_actions"] = parsed_result
                logger.debug(f"Parsed actions for {prefix}: {parsed_result}")
            elif prefix == "final_decision":
                data["parsed_actions"] = json.loads(response)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {prefix} I/O data to {filename}")
        except Exception as e:
            logger.error(f"Error saving {prefix} I/O data to {filename}: {e}", exc_info=True)

    def _save_image(self, image: np.ndarray, directory: str, prefix: str) -> str:
        filename = os.path.join(directory, f"{prefix}_{self.step_count:04d}.png")
        cv2.imwrite(filename, image)
        return filename

    def annotate_units_on_frame(self, frame: np.ndarray, unit_info: List[Dict[str, Any]],
                                unit_numbers: List[int]) -> np.ndarray:
        units_to_annotate = [unit for unit in unit_info if unit['tag_index'] in unit_numbers]
        return _annotate_units_on_image(frame, units_to_annotate)

    def _update_history(self, important_units: List[int], action: Dict[str, List[Tuple]]):
        self.history.append({
            "step": self.step_count,
            "important_units": important_units,
            "attack_actions": action['attack']
        })
        if len(self.history) > self.history_length:
            self.history.pop(0)
