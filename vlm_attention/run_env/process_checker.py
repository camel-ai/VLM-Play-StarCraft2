import psutil
import re




"""

Old version of process checker, not used now.


In old version, we used OBS to record the screen realtime, and then use the process checker to check the process.
But now, we use the pysc2 to record the screen, so the process checker is not used.
"""


def check_processes():
    sc2_processes = []
    all_processes = []

    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            proc_info = proc.info
            all_processes.append(proc_info)

            # 检查进程名称或可执行文件路径是否包含 SC2 或 StarCraft 相关字符串
            if any(sc2_term in proc_info['name'].lower() or
                   (proc_info['exe'] and sc2_term in proc_info['exe'].lower())
                   for sc2_term in ['sc2', 'starcraft']):
                sc2_processes.append(proc_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    print("可能的 StarCraft II 相关进程:")
    for proc in sc2_processes:
        print(f"PID: {proc['pid']}, 名称: {proc['name']}, 可执行文件: {proc['exe']}")

    print("\n所有运行中的进程:")
    for proc in all_processes:
        print(f"PID: {proc['pid']}, 名称: {proc['name']}")

    # 检查与虚拟摄像头相关的进程
    print("\n可能的虚拟摄像头相关进程:")
    for proc in all_processes:
        if any(cam_term in proc['name'].lower() for cam_term in ['cam', 'camera', 'video']):
            print(f"PID: {proc['pid']}, 名称: {proc['name']}")


if __name__ == "__main__":
    check_processes()