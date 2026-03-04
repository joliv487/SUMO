import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.sumo_env import SumoEnv

SUMOCFG = r"C:\Users\joliv\Documents\sumo-ai\scenarios\grid\sim.sumocfg"

def make_env():
    return SumoEnv(SUMOCFG, gui=False)

env = DummyVecEnv([make_env])

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=50000)

model.save("../outputs/traffic_model")

print("Training complete")