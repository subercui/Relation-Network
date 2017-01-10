import numpy as np
import time
import theano
from theano import tensor as T
from layers import GRU, BiGRU
from utils import adadelta, say


class BaseModel(object):
    def __init__(self, args):
        self.args = args

    def train(self, train_tuple, valid_tuple=None):
        grads = T.grad(self.cost, self.params)
        updates = adadelta(self.params, grads)
        self.train_func = theano.function(
            inputs=[self.x, self.y],
            outputs=[self.cost, self.error, self.y_pred],
            updates=updates,
            allow_input_downcast=True
        )
        valid_func = theano.function(
            inputs=[self.x, self.y],
            outputs=[self.cost, self.error, self.y_pred],
            allow_input_downcast=True
        )

        unchanged = 0
        min_cost_before = np.inf
        for epoch in xrange(self.args.max_epochs):
            start = time.time()
            unchanged += 1
            if unchanged > 6: return
            train_batches_x, train_batches_y = self.create_batches(train_tuple)

            train_cost = 0.0
            train_error = 0.0
            abs_error = 0.0
            N = len(train_batches_x)
            for i in xrange(N):
                if (i + 1) % int(N / 10) == 0:
                    etc = time.time() - start
                    say("\r{}/{} ETC:{:.1f}s  ".format(i + 1, N, etc))
                bx, by = train_batches_x[i], train_batches_y[i]
                cost, error, y_pred = self.train_func(bx, by)
                # print y_pred, by
                train_cost += cost
                train_error += error
                abs_error += np.mean(np.abs(y_pred - by))
            train_cost /= N
            train_error /= N
            abs_error /= N
            say("epoch: {}, train_cost is {:.3f}, train error rate is {}, abs error is {}.\n".format(epoch, train_cost,
                                                                                                    train_error,
                                                                                                    abs_error))
            if valid_tuple is not None:
                valid_batches_x, valid_batches_y = self.create_batches(valid_tuple)
                valid_cost = 0.0
                valid_error = 0.0
                valid_abs_error = 0.0
                N = len(valid_batches_x)
                for i in xrange(N):
                    bx, by = valid_batches_x[i], valid_batches_y[i]
                    cost, error, y_pred = valid_func(bx, by)
                    valid_cost += cost
                    valid_error += error
                    valid_abs_error += np.mean(np.abs(y_pred - by))
                valid_cost /= N
                valid_error /= N
                valid_abs_error /= N
                say("\rvalid: cost {:.3f}, error {}, abs_error{}.\n".format(valid_cost, valid_error, valid_abs_error))
            if valid_cost < min_cost_before:
                min_cost_before = valid_cost
                unchanged = 0

    def create_batches(self, data_tuple):
        if isinstance(data_tuple[0], np.ndarray):
            batches_x, batches_y = [], []
            assert len(data_tuple[0]) == len(data_tuple[1])
            bs = self.args.batch_size
            for i in xrange(int(np.ceil(len(data_tuple[0]) / float(bs)))):
                batches_x.append(data_tuple[0][i * bs:i * bs + bs])
                batches_y.append(data_tuple[1][i * bs:i * bs + bs])
            return batches_x, batches_y
        elif isinstance(data_tuple[0], list):
            return data_tuple[0], data_tuple[1]
        else:
            raise NotImplementedError


class HierachEncoder(object):
    def __init__(self, n_in, n_hids, with_contex=False, n_layers=2, func=GRU):
        self.layers = []
        self.params = []
        self.layers.append(func(n_in, n_hids, with_contex))
        self.params += self.layers[0].params
        for i in xrange(1, n_layers):
            self.layers.append(func(n_hids, n_hids, with_contex))
            self.params += self.layers[i].params

    def apply(self, inputs, mask_inputs):
        """
        :param inputs: (length * batch * n_in)
        :return: (batch*n_hids)
        """
        vector = self.layers[0].apply(inputs, mask_inputs)
        for i in xrange(1, len(self.layers)):
            vector = self.layers[i].apply(vector, mask_inputs)

        self.output = vector[-1]
        return self.output
