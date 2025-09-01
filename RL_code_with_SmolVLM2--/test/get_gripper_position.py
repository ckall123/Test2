# play_gripper.py
import time
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from objects.xarm_positions import warm_tf, get_gripper_span, get_arm_bounds


def main():
    rclpy.init()
    node = Node("play_gripper_node")
    tf_buffer = Buffer()
    listener = TransformListener(tf_buffer, node)

    try:
        print("Warming TF (spin_once)...")
        warm_tf(node, tf_buffer, spins=60, timeout_sec=0.02)  # 約 1.2s，視環境調整

        # 1) 拿一次 arm bounds（通常只在 reset 時需要）
        try:
            bounds = get_arm_bounds(tf_buffer, reference_frame="world", margin=0.05)
            print("Arm bounds (world):", bounds)
        except Exception as e:
            print("Could not get arm bounds:", e)

        # 2) 拿一次 gripper span（world）
        try:
            span = get_gripper_span(tf_buffer, reference_frame="world", timeout_sec=0.05)
            print("Initial gripper span:", span)
        except Exception as e:
            print("Could not get initial gripper span:", e)

        # 3) 模擬每 step 讀值：用非阻塞 spin_once 更新 TF buffer
        print("\nSimulating per-step reads (10 steps):")
        for step in range(10):
            # 非阻塞：處理 callback，把新的 /tf 丟進 buffer
            rclpy.spin_once(node, timeout_sec=0.0)

            # 讀 gripper（使用短 timeout 防止卡住）
            try:
                span = get_gripper_span(tf_buffer, reference_frame="world", timeout_sec=0.02)
                print(f"[{step}] gripper z={span['z_center']:.3f}, width={span['width']:.3f}, ref={span['reference_used']}")
            except Exception as e:
                # 若抓不到，safe fallback（假設沒碰到）
                print(f"[{step}] get_gripper_span failed (ok): {e}")

            time.sleep(0.1)  # 模擬 env step 時間

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
