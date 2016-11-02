import numpy as np
from sklearn.preprocessing import StandardScaler


class Actor_Critic(object):

    def __init__(self, env, policy, value, hparams):
        self.hparams = hparams
        self.policy = policy
        self.value = value
        self.env = env
        self.discount = self.hparams["discount"]

        observation_examples = [env.observation_space.sample()
                                for _ in range(10000)]
        self.preprocess = StandardScaler()
        self.preprocess.fit(observation_examples)

    def episode(self, render=False, train=True):

        vloss_list, ploss_list = [], []

        ob = self.env.reset()
        ob = self.preprocess.transform([ob])

        done, iteration = False, 0
        while not done:
            action = self.policy.act(ob)
            next_ob, reward, done, _ = self.env.step([action])
            next_ob = self.preprocess.transform([next_ob])
            reward -= np.abs(self.env.goal_position - self.env.state[0])

            # Calculate TD Targets
            next_value = self.value.estimate(ob)
            if done:
                next_value = np.zeros((1, 1))
                reward -= 100
            value = self.value.estimate(ob)
            td_target = reward + self.discount * next_value
            td_error = td_target - value

            if train:
                # update approximators
                vloss = self.value.learn(ob, td_target)
                ploss = self.policy.learn(ob, td_error)
                vloss_list.append(vloss)
                ploss_list.append(ploss)

            ob = next_ob

            if render:
                self.env.render()

            if iteration % 200 == 0:
                print("Iteration: ", iteration)
            iteration += 1

        print("Episode Length: ", iteration)

        if train:
            return np.array(vloss_list), np.array(ploss_list)
