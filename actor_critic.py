import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import GAE, discount


class Actor_Critic(object):

    def __init__(self, env, policy, value, hparams,
                 reward_fn=None):
        """
        Params:
           gamma: discount factor
           lamb: lambda for generalized adv. estimates
           reward_fn: custom processing of rewards
        """
        self.hparams = hparams
        self.policy = policy
        self.value = value
        self.env = env
        self.reward_fn = reward_fn
        self.gamma = hparams["discount"]
        self.lamb = hparams["lambda"]
        self.iterations = hparams["iterations"]
        self.episodes_per_batch = hparams["episodes_per_batch"]
        self.episode_max_len = hparams["episode_max_len"]

        observation_examples = [env.observation_space.sample()
                                for _ in range(10000)]
        self.preprocess = StandardScaler()
        self.preprocess.fit(observation_examples)

    def do_episode(self, video):
        """Run a single episode."""
        obs = []
        rewards = []
        actions = []
        ob = self.env.reset()
        done = False
        for _ in range(self.episode_max_len):
            obs.append(ob)
            ob = self.preprocess.transform([ob])
            action = self.policy.act(ob)
            actions.append(action)
            ob, reward, done, _ = self.env.step([action])
            if self.reward_fn is not None:
                reward = self.reward_fn(self.env, reward)
            rewards.append(reward)
            if done:
                break
            if video:
                self.env.render()

        return {"rewards": np.array(rewards),
                "obs": np.array(obs),
                "actions": np.array(actions)}

    def collect_trajs(self, video):
        """Run episodes and concatenate data."""
        size = 0
        trajs = []
        lengths = []

        for _ in range(self.episodes_per_batch):
            traj = self.do_episode(video)
            trajs.append(traj)
            length = len(traj["rewards"])
            size += length
            lengths.append(length)

        obs = np.concatenate([traj["obs"] for traj in trajs])
        rewards = np.concatenate([traj["rewards"] for traj in trajs])
        actions = np.concatenate([traj["actions"] for traj in trajs])
        returns = np.concatenate([discount(traj["rewards"], self.gamma)
                                  for traj in trajs])

        return dict(obs=obs, rewards=rewards, actions=actions,
                    returns=returns, lengths=np.array(lengths))

    def train(self, video=False):

        for i in range(self.iterations):
            print("Iteration :", i)
            trajs = self.collect_trajs(video)

            # standardize observations
            obs = self.preprocess.transform(trajs["obs"])

            # Calculate generalized advantage estimates
            next_values = np.squeeze(self.value.estimate(obs))
            values = np.squeeze(self.value.estimate(obs))
            td_targets = trajs["rewards"] + self.gamma * next_values
            td_errors = td_targets - values
            gae_vals = GAE(self.lamb, td_errors,
                           trajs["lengths"]).reshape(-1, 1)

            # train the approximators
            returns = trajs["returns"].reshape(-1, 1)
            actions = trajs["actions"].reshape(-1, 1)
            self.value.learn(obs, returns)
            self.policy.learn(obs, gae_vals, actions)
