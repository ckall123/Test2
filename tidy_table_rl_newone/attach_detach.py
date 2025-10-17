import subprocess

# 預設夾爪設定（根據你的 URDF / plugin 命名）
GRIPPER_MODEL = "UF_ROBOT"
GRIPPER_LINK = "right_finger"
OBJECT_LINK = "link"  # 所有物件的主連接名稱

def attach_object(object_name: str):
    """
    將指定物件附著到夾爪。
    :param object_name: 模型名稱（如 'obj_0'），須與 SDF 中一致。
    """
    cmd = [
        "ros2", "service", "call", "/ATTACHLINK", "linkattacher_msgs/srv/AttachLink",
        f"{{model1_name: '{GRIPPER_MODEL}', link1_name: '{GRIPPER_LINK}', "
        f"model2_name: '{object_name}', link2_name: '{OBJECT_LINK}'}}"
    ]
    subprocess.run(cmd)
    print(f"✅ attach_object: {object_name}")

def detach_object(object_name: str):
    """
    將指定物件從夾爪分離。
    :param object_name: 模型名稱（如 'obj_0'），須與 SDF 中一致。
    """
    cmd = [
        "ros2", "service", "call", "/DETACHLINK", "linkattacher_msgs/srv/DetachLink",
        f"{{model1_name: '{GRIPPER_MODEL}', link1_name: '{GRIPPER_LINK}', "
        f"model2_name: '{object_name}', link2_name: '{OBJECT_LINK}'}}"
    ]
    subprocess.run(cmd)
    print(f"✅ detach_object: {object_name}")


# 🧪 測試
if __name__ == '__main__':
    test_object = "obj_0"
    attach_object(test_object)
    # detach_object(test_object)
