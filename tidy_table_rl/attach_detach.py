import subprocess

# 預設夾爪設定（根據你的 URDF / plugin 命名）
GRIPPER_MODEL = "UF_ROBOT"
GRIPPER_LINK = "right_finger"
OBJECT_LINK = "link"  # 所有物件的主連接名稱

def attach_object(object_name: str) -> bool:
    """
    將指定物件附著到夾爪。
    :param object_name: 模型名稱（如 'obj_0'），須與 SDF 中一致。
    :return: 成功與否
    """
    cmd = [
        "ros2", "service", "call", "/ATTACHLINK", "linkattacher_msgs/srv/AttachLink",
        f"{{model1_name: '{GRIPPER_MODEL}', link1_name: '{GRIPPER_LINK}', "
        f"model2_name: '{object_name}', link2_name: '{OBJECT_LINK}'}}"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
        if result.returncode == 0:
            print(f"✅ attach_object: {object_name}")
            return True
        else:
            print(f"❌ 附著失敗: {object_name}\n{result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ attach_object 發生錯誤: {e}")
        return False

def detach_object(object_name: str) -> bool:
    """
    將指定物件從夾爪分離。
    :param object_name: 模型名稱（如 'obj_0'），須與 SDF 中一致。
    :return: 成功與否
    """
    cmd = [
        "ros2", "service", "call", "/DETACHLINK", "linkattacher_msgs/srv/DetachLink",
        f"{{model1_name: '{GRIPPER_MODEL}', link1_name: '{GRIPPER_LINK}', "
        f"model2_name: '{object_name}', link2_name: '{OBJECT_LINK}'}}"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
        if result.returncode == 0:
            print(f"✅ detach_object: {object_name}")
            return True
        else:
            print(f"❌ 分離失敗: {object_name}\n{result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ detach_object 發生錯誤: {e}")
        return False


# 🧪 測試
if __name__ == '__main__':
    test_object = "coke_can_1"
    if attach_object(test_object):
        print("🎉 測試附著成功")
    if detach_object(test_object):
        print("🎉 測試分離成功")
