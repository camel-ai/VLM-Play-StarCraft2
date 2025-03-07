from pysc2.maps import lib


class VLM_ATTENTION(lib.Map):
  directory = "VLM_ATTENTION"
  download = "https://github.com/Blizzard/s2client-proto#map-packs"
  players = 2
  game_steps_per_episode = 16 * 60 * 30  # 30 minute limit.


vlm_attention_maps = [
  "2c_vs_64zg_vlm_attention",
  "2m_vs_1z_vlm_attention",
  "2s_vs_1sc_vlm_attention",
  "2s3z_vlm_attention",
  "3m_vlm_attention",
  "3s_vs_3z_vlm_attention",
  "6reaper_vs8zealot_vlm_attention",
  "2bc1prism_vs_8m_vlm_attention",
  "8marine_1medvac_vs_2tank",
  "8marine_2tank_vs_zerglings_banelings_vlm_attention",
  "ability_8stalker_vs_8marine_3marauder_1medivac_tank_map",
  "3m_3m_test",
  "vlm_test_map",
  "test_for_cluster",
  "test_for_cluster_version2",
  "test_for_cluster_8_22",
  "vlm_attention_1",
  "vlm_attention_1_two_players",
  "vlm_attention_2_terran_vs_terran_two_players",
  "pvz_task6_level3",
  "MMM_vlm_attention_two_players",
  "ability_test_map",
  "ability_map_8marine_3marauder_1medivac_1tank_2_players",
  "ability_8stalker_vs_8marine_3marauder_1medivac_tank_map_2_players",
  "ability_7stalker_vs_11marine_1medivac_1tank_map_2_players",
  "ability_map_8marine_3marauder_1medivac_1tank"

]

for name in vlm_attention_maps:
  globals()[name] = type(name, (VLM_ATTENTION,), dict(filename=name))