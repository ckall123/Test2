import rclpy
from object.spawner import Spawner
import time

def main():
    rclpy.init()

    # 初始化 Spawner
    spawner = Spawner()

    # 刪除前一輪物體（保險起見）
    spawner.delete_all()

    # 隨機生成 3 個物體（可以改數量）
    print("\n[Spawner] 嘗試產生 3 個物體...\n")
    result = spawner.spawn_random_objects(count=3)

    if not result:
        print("[Spawner] 沒有成功生成任何物體 QQ")
    else:
        print("[Spawner] 成功生成以下物體：")
        for name, pos in result:
            print(f" - {name}: {pos}")

    print("\n[Spawner] 等待 10 秒觀察...")
    time.sleep(10)

    print("[Spawner] 清除所有物體...")
    spawner.delete_all()

    spawner.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
