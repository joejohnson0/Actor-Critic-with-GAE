import tensorflow as tf


class TF_Value(object):
    """Value Function approximator."""

    def __init__(self, net, ob_dim, hparams):
        self.hparams = hparams
        self.graph = tf.Graph()
        self.net = net
        with self.graph.as_default():
            self._init_graph(ob_dim)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def _init_graph(self, ob_dim):

        # inputs
        self.obs = tf.placeholder(dtype=tf.float32, shape=(None, ob_dim))
        self.targets = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        # value estimates
        self.values = self.net.make_net(self.obs)

        self.loss = tf.reduce_mean(tf.square(self.values - self.targets))

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.hparams["learning_rate"])

        self.train_op = self.optimizer.minimize(self.loss)

        self.init_op = tf.initialize_all_variables()

    def estimate(self, obs):
        return self.sess.run(self.values, {self.obs: obs})

    def learn(self, obs, targets):
        feed_dict = {self.obs: obs, self.targets: targets}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss
