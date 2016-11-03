# Actor-Critic-with-GAE

Batch method implementation of the actor-critic method. Two approximators are used: one for the policy, the other for the 
value function. The value approximator learns by estimating returns. The policy network learns using policy gradient with
generalized advantage estimates (Schulman et al 2015). 

It should run with any continuous OpenAI environment. To change the environment or any hyperparams edit the config file.

<b> To-Do </b>: Hyperparameter tuning. Data collection for debugging.
