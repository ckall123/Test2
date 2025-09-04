from my_RL_code.env.tidy_cube_env import TidyCubeEnv  # 換成你實際的檔案名
import time

def main():
    env = TidyCubeEnv(render_mode="human", resolution=(64, 64))
    obs, _ = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.03)  # 額外 sleep 保證畫面穩定不卡卡

    env.close()

if __name__ == "__main__":
    main()
