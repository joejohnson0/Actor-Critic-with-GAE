import gym
from actor import TF_CPolicy
from critic import TF_Value
from actor_critic import Actor_Critic
from approximators import MLP
from config import *

env = gym.envs.make(env)
MAX_ACTION = env.action_space.high
MIN_ACTION = env.action_space.low

ob_dim = env.observation_space.sample().shape[0]
ac_dim = env.action_space.sample().shape[0]

pnet = MLP(2 * ac_dim, pnet_hparams)
vnet = MLP(1, vnet_hparams)

actor = TF_CPolicy(pnet, ob_dim, ac_dim, hparams=actor_hparams,
                   min_val=MIN_ACTION, max_val=MAX_ACTION)

critic = TF_Value(vnet, ob_dim, hparams=critic_hparams)

ac = Actor_Critic(env, actor, critic, hparams=ac_hparams)

for i in range(n_episodes):
    print("Episode Number: ", i + 1)
    train = False
    if i < 5:
        train = True
    ac.episode(train=train, render=True)
