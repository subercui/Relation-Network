import theano
import theano.tensor as T
import numpy, time, cPickle
import logging,sys
from itertools import izip

logger = logging.getLogger(__name__)


def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()


def load_dataset(*filenames):
    data = []
    for filename in filenames:
        if filename.endswith(".pkl"):
            with open(filename, 'r') as f:
                data.append(cPickle.load(f))
        elif filename.endswith(".txt"):
            data.append(numpy.loadtxt(filename, dtype='int64'))
        else:
            raise NotImplementedError
    return data


def save(model, filename):
    say('saving model to {} ...\n'.format(filename))
    params_value = [value.get_value() for value in model.params]
    with open(filename, 'w') as f:
        cPickle.dump(params_value, f, -1)


def load(model, filename):
    say('load model from {} ...\n'.format(filename))
    if filename.endswith('.pkl'):
        with open(filename, 'r') as f:
            params_value = cPickle.load(f)
        assert len(params_value) == len(model.params)
        for i in xrange(len(model.params)):
            model.params[i].set_value(params_value[i])
    else:
        raise NotImplementedError


class param_init(object):

    def __init__(self,**kwargs):

        self.shared = kwargs.pop('shared', True)

    def param(self, size, init_type=None, name=None, **kwargs):
        try:
            if init_type is not None:
                func = getattr(self, init_type)
            elif len(size) == 1:
                func = getattr(self, 'constant')
            elif size[0] == size[1]:
                func = getattr(self, 'orth')
            else:
                func = getattr(self, 'normal')
        except AttributeError:
            logger.error('AttributeError, {}'.format(init_type))
        else:
            param = func(size, name=None, **kwargs)
        return param

    def uniform(self, size, name=None,**kwargs):
        #low = kwargs.pop('low', -6./sum(size))
        #high = kwargs.pop('high', 6./sum(size))
        low = kwargs.pop('low', -0.01)
        high = kwargs.pop('high', 0.01)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.uniform(low=low, high=high, size=size),
            dtype=theano.config.floatX)
        if self.shared:
            param = theano.shared(value=param, borrow=True, name=name)
        return param

    def normal(self, size, name=None,**kwargs):
        loc = kwargs.pop('loc', 0.)
        scale = kwargs.pop('scale', 0.05)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.normal(loc=loc, scale=scale, size=size),
            dtype=theano.config.floatX)
        if self.shared:
            param = theano.shared(value=param, borrow=True, name=name)
        return param

    def constant(self, size, name=None,**kwargs):
        value = kwargs.pop('scale', 0.)
        param = numpy.ones(size, dtype=theano.config.floatX)*value
        if self.shared:
            param = theano.shared(value=param, borrow=True, name=name)
        return param

    def orth(self, size, name=None,**kwargs):
        scale = kwargs.pop('scale', 1.)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        if len(size) != 2:
            raise ValueError
        if size[0] == size[1]:
            M = rng.randn(*size).astype(theano.config.floatX)
            Q, R = numpy.linalg.qr(M)
            Q = Q * numpy.sign(numpy.diag(R))
            param = Q*scale
            if self.shared:
                param = theano.shared(value=param, borrow=True, name=name)
            return param
        else:
            M1 = rng.randn(size[0], size[0]).astype(theano.config.floatX)
            M2 = rng.randn(size[1], size[1]).astype(theano.config.floatX)
            Q1, R1 = numpy.linalg.qr(M1)
            Q2, R2 = numpy.linalg.qr(M2)
            Q1 = Q1 * numpy.sign(numpy.diag(R1))
            Q2 = Q2 * numpy.sign(numpy.diag(R2))
            n_min = min(size[0], size[1])
            param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            if self.shared:
                param = theano.shared(value=param, borrow=True, name=name)
            return param


def  repeat_x(x, n_times):
    # This is black magic based on broadcasting,
    # that's why variable names don't make any sense.
    a = T.shape_padleft(x)
    padding = [1] * x.ndim
    b = T.alloc(numpy.float32(1), n_times, *padding)
    out = a * b
    return out

def  repeat_x_row(x, n_times):
    # This is black magic based on broadcasting,
    # that's why variable names don't make any sense.
    a = T.shape_padaxis
    padding = [1] * x.ndim
    b = T.alloc(numpy.float32(1), n_times, *padding)
    out = a * b
    return out

def adadelta(parameters,gradients,rho=0.95,eps=1e-6):
    # create variables to store intermediate updates
    gradients_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype='float32')) for p in parameters ]
    deltas_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype='float32')) for p in parameters ]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq, g in izip(gradients_sq,gradients)]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    parameters_updates = [ (p,T.clip(p - d, -40, 40)) for p,d in izip(parameters,deltas) ]
    return gradient_sq_updates + deltas_sq_updates + parameters_updates

def create_index2word(word2index_dict):
    index2word = [0]*len(word2index_dict)
    for key, value in word2index_dict.iteritems():
        index2word[value] = key
    return index2word
    
def read_word2vec():
    f = open('/home/shalei/word2vec/data/text8vectors.txt')
    lines = f.readlines()[1:]
    g = lambda x : numpy.array(x, dtype = numpy.double)
    return {line.strip().split(' ')[0]:g(line.strip().split(' ')[1:]) for line in lines}

if __name__ == "__main__":
    Wemb = read_word2vec()
    Wembkey = [a for a in Wemb]
    Wembid = numpy.asarray([Wemb[a] for a in Wembkey])
    tic = time.time()
    Wembkeypos = {a:Wembkey.index(a) for a in Wembkey}
    toc = time.time()
    print toc - tic, 's'
