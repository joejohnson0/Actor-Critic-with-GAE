env = "MountainCarContinuous-v0"
#env = "Pendulum-v0"

n_episodes = 10

pnet_hparams = {"n_hlayers": 3, "n_hidden": [30, 20, 10]}

vnet_hparams = {"n_hlayers": 3, "n_hidden": [30, 20, 10]}

actor_hparams = {"learning_rate": 0.001}

critic_hparams = {"learning_rate": 0.001}

ac_hparams = {"discount": 0.9}
