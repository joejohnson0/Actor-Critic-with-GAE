import tensorflow as tf
import numpy as np


GAUSS_FACTOR = 1.0 / np.sqrt(2*np.pi)


class TF_CPolicy(object):
    """Continuous Neural Network Policy Function in Tensorflow."""

    def __init__(self, net, ob_dim, ac_dim, hparams, min_val, max_val):
        self.hparams = hparams
        self.min_val = min_val
        self.max_val = max_val
        self.net = net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._init_graph(ob_dim, ac_dim)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def _init_graph(self, ob_dim, ac_dim):

        # inputs
        self.ob = tf.placeholder(dtype=tf.float32, shape=(None, ob_dim))
        self.target = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        # get gaussian params from network and clip
        self.net_out = self.net.make_net(self.ob)
        self.mu = tf.squeeze(self.net_out[:, :ac_dim])
        self.mu = tf.clip_by_value(self.mu, self.min_val, self.max_val)
        self.sigma = tf.squeeze(tf.exp(self.net_out[:, ac_dim:]))
        self.sigma = tf.clip_by_value(self.sigma, 1, 3)

        # evaluate action probs with the gaussian distributions
        self.normal = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.normal.sample_n(1)
        self.action = tf.clip_by_value(self.action, self.min_val, self.max_val)

        # Loss and training
        self.loss = -self.normal.log_prob(self.action) * self.target
        self.loss -= self.normal.entropy()

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hparams["learning_rate"])
        self.train_op = self.optimizer.minimize(self.loss)

        self.init_op = tf.initialize_all_variables()

    def act(self, ob):
        return np.squeeze(self.sess.run([self.action], {self.ob: ob}))

    def learn(self, obs, targets):
        feed_dict = {self.ob: obs, self.target: targets}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss
