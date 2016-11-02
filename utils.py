import numpy as np


def apply_discount(x, gamma):
    out = np.zeros(len(x), dtype=np.float64)
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    return out


def permute(list_data):
    permutation = np.random.permutation(list_data[0].shape[0])
    for data in list_data:
        data[...] = data[permutation]
    return list_data


def do_episode(env, actor, episode_max_length,
               render=True):
    obs = []
    rewards = []
    actions = []
    ob = env.reset()
    done = False
    for _ in range(episode_max_length):
        obs.append(ob)
        action = actor.act(ob)
        actions.append(action)
        ob, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
        if render:
            env.render()

    return {"rewards": np.array(rewards),
            "obs": np.array(obs),
            "actions": np.array(actions)}


def collect_traj(timesteps):
    size = 0
    trajs = []
    # collect enough trajectories
    while size < timesteps:
        traj = do_episode()
        trajs.append(traj)
        size += len(traj["rewards"])

    # concatenate into numpy arrays
    obs = np.concatenate([traj["obs"] for traj in trajs])
    obs = obs[:timesteps]

    rewards = np.concatenate([traj["rewards"] for traj in trajs])
    rewards = rewards[:timesteps]

    actions = [traj["actions"] for traj in trajs]
    actions = np.concatenate(actions)[:timesteps]
    
    return {"obs": obs, "actions": actions, "rewards": rewards}
