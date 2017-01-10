import theano
import theano.tensor as T
import numpy
import nonlinearities
from utils import param_init, repeat_x
from theano.tensor.nnet import categorical_crossentropy


def _p(pp, name):
    return '%s_%s' % (pp, name)


class DenseLayer(object):
    """
    author: Cui Haotian
    """

    def __init__(self, n_in, n_out, nonlinearity=nonlinearities.rectify, prefix='Dense'):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = param_init().uniform((n_in, n_out), name=_p(prefix, 'W'))
        # initialize the baises b as a vector of n_out 0s
        self.b = param_init().constant((n_out,), name=_p(prefix, 'b'))
        self.params = [self.W, self.b]
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

    def get_output_for(self, input):
        assert input.ndim == 2

        return self.nonlinearity(T.dot(input, self.W) + self.b)


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, prefix='Logist'):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = param_init().uniform((n_in, n_out), name=_p(prefix, 'W'))
        # initialize the baises b as a vector of n_out 0s
        self.b = param_init().constant((n_out,), name=_p(prefix, 'b'))

        # compute vector of class-membership probabilities in symbolic form
        energy = theano.dot(input, self.W) + self.b
        if energy.ndim == 3:
            energy_exp = T.exp(energy - T.max(energy, 2, keepdims=True))
            pmf = energy_exp / energy_exp.sum(2, keepdims=True)
        else:
            pmf = T.nnet.softmax(energy)

        self.p_y_given_x = pmf
        self.y_pred = T.argmax(self.p_y_given_x, axis=-1)

        # compute prediction as class whose probability is maximal in
        # symbolic form

        # parameters of the model
        self.params = [self.W, self.b]

    def cost(self, targets, mask=None):
        prediction = self.p_y_given_x

        if prediction.ndim == 3:
            # prediction = prediction.dimshuffle(1,2,0).flatten(2).dimshuffle(1,0)
            prediction_flat = prediction.reshape(((prediction.shape[0] *
                                                   prediction.shape[1]),
                                                  prediction.shape[2]), ndim=2)
            targets_flat = targets.flatten()
            mask_flat = mask.flatten()
            ce = categorical_crossentropy(prediction_flat, targets_flat) * mask_flat
        else:
            ce = categorical_crossentropy(prediction, targets)
        return T.sum(ce)

        # liuxianggen

    def cost_entry(self, targets, mask=None):
        prediction = self.p_y_given_x  # (9,5,24)

        if prediction.ndim == 3:
            # prediction = prediction.dimshuffle(1,2,0).flatten(2).dimshuffle(1,0)
            prediction_flat = prediction.reshape(((prediction.shape[0] *
                                                   prediction.shape[1]),
                                                  prediction.shape[2]), ndim=2)  # (45,24)
            targets_flat = targets.flatten()
            mask_flat = mask.flatten()
            ce = categorical_crossentropy(prediction_flat, targets_flat) * mask_flat
        else:
            ce = categorical_crossentropy(prediction, targets)
        ce_entry = ce.reshape((prediction.shape[0], prediction.shape[1]), ndim=2).sum(axis=0)  # (5)
        return ce_entry

    def errors(self, y):
        y_pred = self.y_pred
        if y.ndim == 2:
            y = y.flatten()
            y_pred = y_pred.flatten()
        return T.sum(T.neq(y, y_pred))


class GRU(object):
    def __init__(self, n_in, n_hids, with_contex=False, **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self._init_params()

    def _init_params(self):
        n_in = self.n_in
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)  # (30,39)
        size_hh = (n_hids, n_hids)  # (39,39)
        self.W_xz = param_init().uniform(size_xh)  # (30,39)
        self.W_xr = param_init().uniform(size_xh)  # (30,39)
        self.W_xh = param_init().uniform(size_xh)  # (30,39)

        self.W_hz = param_init().orth(size_hh)
        self.W_hr = param_init().orth(size_hh)
        self.W_hh = param_init().orth(size_hh)

        self.b_z = param_init().constant((n_hids,))
        self.b_r = param_init().constant((n_hids,))
        self.b_h = param_init().constant((n_hids,))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        if self.with_contex:
            size_ch = (self.c_hids, self.n_hids)
            self.W_cz = param_init().uniform(size_ch)
            self.W_cr = param_init().uniform(size_ch)
            self.W_ch = param_init().uniform(size_ch)
            self.W_c_init = param_init().uniform(size_ch)

            self.params = self.params + [self.W_cz, self.W_cr,
                                         self.W_ch, self.W_c_init]

    def _step_forward_with_context(self, x_t, x_m, h_tm1, c_z, c_r, c_h):
        """
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        """
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                             T.dot(h_tm1, self.W_hz) + c_z + self.b_z)

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                             T.dot(h_tm1, self.W_hr) + c_r + self.b_r)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) + c_h +
                         self.b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
        return h_t

    def _step_forward(self, x_t, x_m, h_tm1):
        """
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        """
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                             T.dot(h_tm1, self.W_hz) + self.b_z)

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                             T.dot(h_tm1, self.W_hr) + self.b_r)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) +
                         self.b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
        return h_t

    def apply(self, state_below, mask_below, init_state=None, context=None):
        if state_below.ndim == 3:
            # e.g. state_below=(n_step 10, batch_size 5, vector_size 30)
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]

        else:
            raise NotImplementedError

        if mask_below == None:
            mask_below = T.ones(state_below.shape[:2], dtype='float32')
            # mask_below = T.ones_like(state_below,'float32')
            # print mask_below
        if self.with_contex:
            if init_state is None:
                init_state = T.tanh(theano.dot(context, self.W_c_init))
            c_z = theano.dot(context, self.W_cz)
            c_r = theano.dot(context, self.W_cr)
            c_h = theano.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]
            rval, updates = theano.scan(self._step_forward_with_context,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        else:
            if init_state is None:
                # init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                # here if the batch_size = 1, it will meet the error:
                # "Inconsistency in the inner graph of scan 'scan_fn' : an input and an output are associated with the same recurrent state and should have the same type but have type 'TensorType(float32, row)' and 'TensorType(float32, matrix)' respectively.")
                # so I correct this line to the below
                init_state = T.unbroadcast(T.alloc(numpy.float32(0.), batch_size, self.n_hids), 0)
            rval, updates = theano.scan(self._step_forward,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        n_steps=n_steps
                                        )
        self.output = rval
        return self.output

    def merge_out(self, state_below, mask_below, context=None):
        hiddens = self.apply(state_below, mask_below, context=context)
        if context is None:
            msize = self.n_in + self.n_hids
            osize = self.n_hids
            combine = T.concatenate([state_below, hiddens], axis=2)
        else:
            msize = self.n_in + self.n_hids + self.c_hids
            osize = self.n_hids
            n_times = state_below.shape[0]
            m_context = repeat_x(context, n_times)
            combine = T.concatenate([state_below, hiddens, m_context], axis=2)

        self.W_m = param_init().uniform((msize, osize * 2))
        self.b_m = param_init().constant((osize * 2,))
        self.params += [self.W_m, self.b_m]

        merge_out = theano.dot(combine, self.W_m) + self.b_m
        merge_max = merge_out.reshape((merge_out.shape[0],
                                       merge_out.shape[1],
                                       merge_out.shape[2] / 2,
                                       2), ndim=4).max(axis=3)
        return merge_max * mask_below[:, :, None]


class DynamicMemory(object):

    def __init__(self, emb_size, hid_size, n_blocks, w_init=None, prefix='DyMem'):
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.n_blocks = n_blocks
        if w_init is None:
            self.w = param_init().uniform((n_blocks, emb_size), name=_p(prefix, 'w'))
        else:
            self.w = w_init
        self.U = param_init().uniform((hid_size, hid_size), name=_p(prefix, 'U'))
        self.V = param_init().uniform((emb_size, hid_size), name=_p(prefix, 'V'))
        self.W = param_init().uniform((emb_size, hid_size), name=_p(prefix, 'W'))
        self.params = [self.w, self.U, self.V, self.W]

    def _step_forward(self, s_t, h_tm1):
        """
        s_t
        h_tm1
        """
        g = T.nnet.sigmoid(s_t*)

    def apply(self, sentences, init_hid=None):
        """
        Parameters
        ----------
        sentences: (length, batch, featuresdim)

        Returns
        -------
        hs: (n_blocks, batch, hid_size)
        """
        if sentences.ndim == 3:
            batch_size = sentences.shape[1]
            n_steps = sentences.shape[0]
        else:
            raise NotImplementedError

        if init_hid is None:
            init_hid = T.unbroadcast(T.alloc(numpy.float32(0.), batch_size, self.hid_size))
        rval, updates = theano.scan(self._step_forward,
                                    sequences=[sentences],
                                    outputs_info=[init_hid],
                                    n_steps=n_steps
                                    )
        self.hs = rval
        return self.hs


class lookup_table(object):
    def __init__(self, embsize, vocab_size):
        self.W = param_init().uniform((vocab_size, embsize))
        self.params = [self.W]
        self.vocab_size = vocab_size
        self.embsize = embsize

    def apply(self, indices):
        outshape = [indices.shape[i] for i
                    in range(indices.ndim)] + [self.embsize]  # ()

        return self.W[indices.flatten()].reshape(outshape)


class auto_encoder(object):
    def __init__(self, sentence, sentence_mask, vocab_size, n_in, n_hids, **kwargs):
        layers = []

        # batch_size = sentence.shape[1]
        encoder = GRU(n_in, n_hids, with_contex=False)
        layers.append(encoder)

        if 'table' in kwargs:
            table = kwargs['table']
        else:
            table = lookup_table(n_in, vocab_size)
        # layers.append(table)

        state_below = table.apply(sentence)
        context = encoder.apply(state_below, sentence_mask)[-1]

        decoder = GRU(n_in, n_hids, with_contex=True)
        layers.append(decoder)

        decoder_state_below = table.apply(sentence[:-1])
        hiddens = decoder.merge_out(decoder_state_below,
                                    sentence_mask[:-1], context=context)

        logistic_layer = LogisticRegression(hiddens, n_hids, vocab_size)
        layers.append(logistic_layer)

        self.cost = logistic_layer.cost(sentence[1:],
                                        sentence_mask[1:]) / sentence_mask[1:].sum()
        self.cost_entry = logistic_layer.cost_entry(sentence[1:],  # (9,5)
                                                    sentence_mask[1:])  # predict model cost, (5)
        self.output = context
        self.params = []
        for layer in layers:
            self.params.extend(layer.params)


class BiGRU(object):
    def __init__(self, n_in, n_hids, with_contex=False, prefix='BiGRU', **kwargs):
        kwargs['merge'] = False
        self.encoder = GRU(n_in, n_hids, with_contex=with_contex, prefix=_p(prefix, 'l2r'), **kwargs)
        self.rencoder = GRU(n_in, n_hids, with_contex=with_contex, prefix=_p(prefix, 'r2l'), **kwargs)

        self.params = self.encoder.params + self.rencoder.params

    def apply(self, state_below, mask):
        rstate_below = state_below[::-1]
        if mask is None:
            rmask = None
        else:
            rmask = mask[::-1]
        loutput = self.encoder.apply(state_below, mask)
        routput = self.rencoder.apply(rstate_below, rmask)

        self.output = T.concatenate([loutput, routput[::-1]], axis=2)
        return self.output


class Attention(object):
    def __init__(self, s_in, t_in, prefix='Attention', **kwargs):
        self.params = []
        self.s_in = s_in
        self.t_in = t_in
        self.align_size = t_in
        self.prefix = prefix

        self.Wa = param_init().param((self.t_in, self.align_size), name=_p(prefix, 'Wa'))
        # self.v = param_init().param((self.align_size,), init_type='constant',
        # name=_p(prefix, 'v'), scale=0.001)

        self.v = param_init().param((self.align_size,), name=_p(prefix, 'v'))
        self.params += [self.Wa, self.v]

    def apply(self, source, source_mask=None, source_x=None, attention=None):
        """

        :param source: the input tensor you want put attention on; shape (length, batch, 'embedding_len or feature_len')
        :param source_mask: mask (length, batch)
        :param source_x: this is the (Ua * h_j)
        :param attention: this is the si-1 in the original paper, dynamic
        :return: 2d (batch, 'embedding_len or feature_len')
        """
        # attention is 2
        if source.ndim != 3 or attention.ndim != 2:
            raise NotImplementedError

        align_matrix = T.tanh(source_x + T.dot(attention, self.Wa)[None, :, :])
        align = theano.dot(align_matrix, self.v)
        align = T.exp(align - align.max(axis=0, keepdims=True))
        # my note: align is the attention scores, like [0.1, 0.2, 0.4, 0.3]
        if source_mask:
            align = align * source_mask
            normalization = align.sum(axis=0) + T.all(1 - source_mask, axis=0)
        else:
            normalization = align.sum(axis=0)
        align = align / normalization
        self.output = (T.shape_padright(align) * source).sum(axis=0)

        return self.output


class SimpleAttention(object):
    def __init__(self, s_in, n_task, prefix='Attention', **kwargs):
        self.s_in = s_in
        self.align_size = s_in
        self.Ws = param_init().param((self.s_in, self.align_size), name=_p(prefix, 'Ws'))
        self.bs = param_init().param((self.align_size,), name=_p(prefix, 'bs'))
        self.v = param_init().param((n_task, self.align_size), name=_p(prefix, 'v'))
        self.params = [self.Ws, self.bs, self.v]

    def apply(self, source, tag):
        if source.ndim != 3:
            raise NotImplementedError

        source_x = T.dot(source, self.Ws) + self.bs
        align_matrix = T.tanh(source_x)
        align = T.dot(align_matrix, self.v[tag])
        align = T.exp(align - align.max(axis=0, keepdims=True))
        normalization = align.sum(axis=0)
        # shape is (length, batch)
        self.align = align / normalization
        self.output = (T.shape_padright(self.align) * source).sum(axis=0)
        return self.output

    def top_n_align(self, n):
        """
        :param n:
        :return: the top n relevant aglign index and respect align score, and the minimum align
        """
        align_index = T.argsort(self.align, axis=0)
        align_order = T.sort(self.align, axis=0)
        return align_index[-n:], align_order[align_index[-n:], T.arange(align_order.shape[1])], align_order[0, :]
