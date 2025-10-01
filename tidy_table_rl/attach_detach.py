import subprocess


def attach_object(object: str):
    """
    以 service call 將物件附著到右夾爪（right_finger）。
    object: 模型名稱（model2_name），須與 SDF 中一致。
    """
    cmd = [
        "ros2", "service", "call", "/ATTACHLINK", "linkattacher_msgs/srv/AttachLink",
        f"{{model1_name: 'UF_ROBOT', link1_name: 'right_finger', model2_name: '{object}', link2_name: 'link'}}"
    ]
    subprocess.run(cmd)
    print(f"✅ attach_object: {object}")


def detach_object(object: str):
    """
    將物件從右夾爪解除附著。
    object: 模型名稱（model2_name），須與 SDF 中一致。
    """
    cmd = [
        "ros2", "service", "call", "/DETACHLINK", "linkattacher_msgs/srv/DetachLink",
        f"{{model1_name: 'UF_ROBOT', link1_name: 'right_finger', model2_name: '{object}', link2_name: 'link'}}"
    ]
    subprocess.run(cmd)
    print(f"✅ detach_object: {object}")


if __name__ == '__main__':
    test_object = "wood_cube_5cm"
    # attach_object(test_object)
    detach_object(test_object)
