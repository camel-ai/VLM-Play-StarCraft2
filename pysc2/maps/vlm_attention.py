from pysc2.maps import lib


class VLM_ATTENTION(lib.Map):
  directory = "VLM_ATTENTION"
  download = "https://github.com/Blizzard/s2client-proto#map-packs"
  players = 2
  game_steps_per_episode = 16 * 60 * 30  # 30 minute limit.


vlm_attention_maps = [
    "1c3s5z",
  "2c_vs_64zg_vlm_attention",
  "2m_vs_1z_vlm_attention",
  "2s_vs_1sc_vlm_attention",
  "2s3z_vlm_attention",
  "3m_vlm_attention",
  "3s_vs_3z_vlm_attention",
  "3s_vs_4z",
  "3s_vs_5z",
  "3s5z",
  "3s5z_vs_3s6z",
  "5m_vs_6m",
  "6h_vs_8z",
  "8m",
  "8m_vs_9m",
  "10m_vs_11m",
  "25m",
  "27m_vs_30m",
  "bane_vs_bane",
  "corridor",
  "MMM",
  "MMM2",
  "so_many_baneling",
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
  "ability_test_map"

]

for name in vlm_attention_maps:
  globals()[name] = type(name, (VLM_ATTENTION,), dict(filename=name))