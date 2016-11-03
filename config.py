env_name = "MountainCarContinuous-v0"
#env_name = "Pendulum-v0"

pnet_hparams = {"n_hlayers": 3, "n_hidden": [30, 20, 10]}

vnet_hparams = {"n_hlayers": 3, "n_hidden": [30, 20, 10]}

actor_hparams = {"learning_rate": 0.001}

critic_hparams = {"learning_rate": 0.001}

ac_hparams = {"discount": 1, "lambda": 0.9,
              "episodes_per_batch": 2,
              "iterations": 10,
              "episode_max_len": 3000}
