import subprocess

'''
model1_name：第一個模型的名稱，通常為機器人本體（robot base），要與你在 URDF 或 SDF 中定義的 model 名稱完全相符。  
link1_name：這個模型中要附著物件的 末端執行器連結（end-effector link），也就是你想讓物件黏上的那個 link 名稱。  
model2_name：第二個模型的名稱，通常是你要附著的 物件，同樣要與 URDF/SDF 中的 model 名稱一致。  
link2_name：這個物件本身的連結名稱，也就是這個 model 裡面哪個 link 要被 attach（通常就是物件的主 link）。  
'''

def attach():
    '''
    model1_name：第一個模型的名稱，通常為機器人本體（robot base），要與你在 URDF 或 SDF 中定義的 model 名稱完全相符。  
    link1_name：這個模型中要附著物件的 末端執行器連結（end-effector link），也就是你想讓物件黏上的那個 link 名稱。  
    model2_name：第二個模型的名稱，通常是你要附著的 物件，同樣要與 URDF/SDF 中的 model 名稱一致。  
    link2_name：這個物件本身的連結名稱，也就是這個 model 裡面哪個 link 要被 attach（通常就是物件的主 link）。  
    '''
    subprocess.run(["ros2", "service", "call", "/ATTACHLINK", "linkattacher_msgs/srv/AttachLink",
                    "{model1_name: 'UF_ROBOT', link1_name: 'right_finger', model2_name: 'test', link2_name: 'link'}"])
    
    print("attach_object")

def detach():
    '''
    model1_name：第一個模型的名稱，通常為機器人本體（robot base），要與你在 URDF 或 SDF 中定義的 model 名稱完全相符。  
    link1_name：這個模型中要附著物件的 末端執行器連結（end-effector link），也就是你想讓物件黏上的那個 link 名稱。  
    model2_name：第二個模型的名稱，通常是你要附著的 物件，同樣要與 URDF/SDF 中的 model 名稱一致。  
    link2_name：這個物件本身的連結名稱，也就是這個 model 裡面哪個 link 要被 attach（通常就是物件的主 link）。  
    '''
    subprocess.run(["ros2", "service", "call", "/DETACHLINK", "linkattacher_msgs/srv/DetachLink",
                    "{model1_name: 'UF_ROBOT', link1_name: 'right_finger', model2_name: 'test', link2_name: 'link'}"])
    
    print("detach_object")



if __name__ == '__main__':
    # attach_object()
    detach()