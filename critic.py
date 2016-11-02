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
        self.ob = tf.placeholder(dtype=tf.float32, shape=(None, ob_dim))
        self.target = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        # value estimates
        self.value = self.net.make_net(self.ob)

        self.loss = tf.reduce_mean(tf.square(self.value - self.target))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hparams["learning_rate"])

        self.train_op = self.optimizer.minimize(self.loss)

        self.init_op = tf.initialize_all_variables()

    def estimate(self, ob):
        return self.sess.run(self.value, {self.ob: ob})

    def learn(self, obs, targets):
        feed_dict = {self.ob: obs, self.target: targets}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss
