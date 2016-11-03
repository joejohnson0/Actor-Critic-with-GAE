import numpy as np
import gym
from actor import TF_CPolicy
from critic import TF_Value
from actor_critic import Actor_Critic
from approximators import MLP
from config import *

# make the environemnt and get its dimensions
env = gym.envs.make(env_name)
MAX_ACTION = env.action_space.high
MIN_ACTION = env.action_space.low

ob_dim = env.observation_space.sample().shape[0]
ac_dim = env.action_space.sample().shape[0]

# MLP function approximators
pnet = MLP(2 * ac_dim, pnet_hparams)
vnet = MLP(1, vnet_hparams)

# actor and critic networks/training graphs in TF
actor = TF_CPolicy(pnet, ob_dim, ac_dim, hparams=actor_hparams,
                   min_val=MIN_ACTION, max_val=MAX_ACTION)
critic = TF_Value(vnet, ob_dim, hparams=critic_hparams)

# change structure of reward fn for car env
if env_name == "MountainCarContinuous-v0":
    def distance_reward(env, reward):
        return reward - np.abs(env.goal_position - env.state[0])
    reward_fn = distance_reward
else:
    reward_fn = None

# train and run actor critic
ac = Actor_Critic(env, actor, critic, hparams=ac_hparams,
                  reward_fn=reward_fn)
ac.train(video=False)
for _ in range(5):
    ac.do_episode(video=True)
