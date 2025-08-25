import subprocess

# 透過 ros2 service 呼叫 Link Attacher（與你的既有實作一致）

def attach(model1: str, link1: str, model2: str, link2: str):
    try:
        subprocess.run([
            "ros2", "service", "call", "/ATTACHLINK", "linkattacher_msgs/srv/AttachLink",
            f"{{model1_name: '{model1}', link1_name: '{link1}', model2_name: '{model2}', link2_name: '{link2}'}}"
        ], check=False)
    except Exception:
        pass


def detach(model1: str, link1: str, model2: str, link2: str):
    try:
        subprocess.run([
            "ros2", "service", "call", "/DETACHLINK", "linkattacher_msgs/srv/DetachLink",
            f"{{model1_name: '{model1}', link1_name: '{link1}', model2_name: '{model2}', link2_name: '{link2}'}}"
        ], check=False)
    except Exception:
        pass