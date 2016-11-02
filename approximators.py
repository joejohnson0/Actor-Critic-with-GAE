import tensorflow as tf


class MLP(object):

    def __init__(self, output_dim, hparams):
        self.n_hlayers = hparams["n_hlayers"]
        self.n_hidden = hparams["n_hidden"]
        assert self.n_hlayers == len(self.n_hidden)
        self.output_dim = output_dim

    def make_net(self, input_op):

        # make the hidden layers
        inputs = input_op
        for i in range(self.n_hlayers):
            hidden = self.make_dense(inputs, self.n_hidden[i])
            hidden = tf.nn.relu(hidden)
            inputs = hidden
        # make output layer
        self.output = self.make_dense(inputs, self.output_dim)
        return self.output

    def make_dense(self, inputs, output_dim):
        input_dim = inputs.get_shape().as_list()[1]
        W = tf.Variable(tf.truncated_normal([input_dim, output_dim]))
        b = tf.Variable(tf.zeros(output_dim))
        hidden = tf.matmul(inputs, W) + b
        return hidden
