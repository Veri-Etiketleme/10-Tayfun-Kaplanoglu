import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
import numpy as np

rng = np.random.RandomState(1234)

def relu(x):
    return T.maximum(0, x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


class Metric(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def negative_log_likelihood(self):
        self.prob_of_y_given_x = T.nnet.softmax(self.x)
        return -T.mean(T.log(self.prob_of_y_given_x)[T.arange(self.y.shape[0]), self.y])

    def cross_entropy(self):
        self.prob_of_y_given_x = T.nnet.softmax(self.x)
        return T.mean(T.nnet.categorical_crossentropy(self.prob_of_y_given_x, self.y))

    def mean_squared_error(self):
        return T.mean((self.x - self.y) ** 2)

    def errors(self):
        if self.y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', self.y.type, 'y_pred', self.y_pred.type))

        if self.y.dtype.startswith('int'):
            self.prob_of_y_given_x = T.nnet.softmax(self.x)
            self.y_pred = T.argmax(self.prob_of_y_given_x, axis=1)
            return T.mean(T.neq(self.y_pred, self.y))
        else:
            return NotImplementedError()

    def accuracy(self):
        if self.y.dtype.startswith('int'):
            self.prob_of_y_given_x = T.nnet.softmax(self.x)
            self.y_pred = T.argmax(self.prob_of_y_given_x, axis=1)
            return T.mean(T.eq(self.y_pred, self.y))
        else:
            return NotImplementedError()


def shared_data(x, y):
    shared_x = theano.shared(
        np.asarray(x, dtype=theano.config.floatX), borrow=True)
    if y is None:
        return shared_x

    shared_y = theano.shared(
        np.asarray(y, dtype=theano.config.floatX), borrow=True)

    return shared_x, T.cast(shared_y, 'int32')


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX),
        name=name,
        borrow=True
    )


def dropout(x, train, p=0.5, rng = np.random.RandomState(1234)):
    masked_x = None
    if p > 0.0 and p < 1.0:
        seed = rng.randint(2 ** 30)
        srng = T.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(
            n=1,
            p=1.0 - p,
            size=x.shape,
            dtype=theano.config.floatX
        )
        masked_x = x * mask
    else:
        masked_x = x
    return T.switch(T.neq(train, 0), masked_x, x * (1.0 - p))


class Optimizer(object):

    def __init__(self, params=None):
        if params is None:
            return NotImplementedError()
        self.params = params

    def updates(self, loss=None):
        if loss is None:
            return NotImplementedError()

        self.updates = OrderedDict()
        self.gparams = [T.grad(loss, param) for param in self.params]


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX),
        name=name,
        borrow=True
    )


class RMSprop(Optimizer):

    def __init__(self, learning_rate=0.001, alpha=0.99, eps=1e-8, params=None):
        super(RMSprop, self).__init__(params=params)

        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps

        self.mss = [
            build_shared_zeros(t.shape.eval(), 'ms') for t in self.params]

    def updates(self, loss=None):
        super(RMSprop, self).updates(loss=loss)

        for ms, param, gparam in zip(self.mss, self.params, self.gparams):
            _ms = ms * self.alpha
            _ms += (1 - self.alpha) * gparam * gparam
            self.updates[ms] = _ms
            self.updates[param] = param - self.learning_rate * \
                gparam / T.sqrt(_ms + self.eps)

        return self.updates

class AdaDelta(Optimizer):

    def __init__(self, rho=0.95, eps=1e-6, params=None):
        super(AdaDelta, self).__init__(params=params)

        self.rho = rho
        self.eps = eps
        self.accugrads = [
            build_shared_zeros(t.shape.eval(), 'accugrad') for t in self.params]
        self.accudeltas = [
            build_shared_zeros(t.shape.eval(), 'accudelta') for t in self.params]

    def updates(self, loss=None):
        super(AdaDelta, self).updates(loss=loss)

        for accugrad, accudelta, param, gparam\
                in zip(self.accugrads, self.accudeltas, self.params, self.gparams):
            agrad = self.rho * accugrad + (1 - self.rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self.eps) / (agrad + self.eps)) * gparam
            self.updates[accudelta] = (
                self.rho * accudelta + (1 - self.rho) * dx * dx)
            self.updates[param] = param + dx
            self.updates[accugrad] = agrad

        return self.updates

class MomentumSGD(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0.9, params=None):
        super(MomentumSGD, self).__init__(params=params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vs = [build_shared_zeros(t.shape.eval(), 'v')
                   for t in self.params]

    def updates(self, loss=None):
        super(MomentumSGD, self).updates(loss=loss)

        for v, param, gparam in zip(self.vs, self.params, self.gparams):
            _v = v * self.momentum
            _v = _v - self.learning_rate * gparam
            self.updates[param] = param + _v
            self.updates[v] = _v

        return self.updates    

class Adam(Optimizer):

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, gamma=1 - 1e-8, params=None):
        super(Adam, self).__init__(params=params)

        self.alpha = alpha
        self.b1 = beta1
        self.b2 = beta2
        self.gamma = gamma
        self.t = theano.shared(np.float32(1))
        self.eps = eps

        self.ms = [build_shared_zeros(t.shape.eval(), 'm')
                   for t in self.params]
        self.vs = [build_shared_zeros(t.shape.eval(), 'v')
                   for t in self.params]

    def updates(self, loss=None):
        super(Adam, self).updates(loss=loss)
        self.b1_t = self.b1 * self.gamma ** (self.t - 1)

        for m, v, param, gparam \
                in zip(self.ms, self.vs, self.params, self.gparams):
            _m = self.b1_t * m + (1 - self.b1_t) * gparam
            _v = self.b2 * v + (1 - self.b2) * gparam ** 2

            m_hat = _m / (1 - self.b1 ** self.t)
            v_hat = _v / (1 - self.b2 ** self.t)

            self.updates[param] = param - self.alpha * \
                m_hat / (T.sqrt(v_hat) + self.eps)
            self.updates[m] = _m
            self.updates[v] = _v
        self.updates[self.t] = self.t + 1.0

        return self.updates

# Multi Layer Perceptron

class Layer:
    # Constructor
    def __init__(self, in_dim, out_dim):
        rng = np.random.RandomState(1234)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = theano.shared(rng.uniform(low=-0.08, high=0.08,
                                           size=(in_dim, out_dim)
                                           ).astype('float32'), name='W')
        self.b = theano.shared(np.zeros(out_dim).astype('float32'), name='b')
        self.params = [self.W, self.b]
        

    # Forward Propagation
    def f_prop(self, x):
        self.z = T.dot(x, self.W) + self.b
        return self.z

class Activation:
    # Constructor
    def __init__(self, function):
        self.function = function
        self.params = []

    # Forward Propagation
    def f_prop(self, x):
        self.z = self.function(x)
        return self.z
    
class BatchNorm:
    # Constructor
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        self.shape = shape
        self.epsilon = epsilon

        self.gamma = theano.shared(np.ones(self.shape, dtype="float32"),
                                   name="gamma")
        self.beta = theano.shared(np.zeros(self.shape, dtype="float32"),
                                  name="beta")
        self.params = [self.gamma, self.beta]

    # Forward Propagation
    def f_prop(self, x):
        if x.ndim == 2:
            mean = T.mean(x, axis=0, keepdims=True)
            std = T.sqrt(T.var(x, axis=0, keepdims=True) + self.epsilon)
        elif x.ndim == 4:
            mean = T.mean(x, axis=(0, 2, 3), keepdims=True)
            std = T.sqrt(T.var(x, axis=(0, 2, 3), keepdims=True) +
                         self.epsilon)

        normalized_x = (x - mean) / std
        self.z = self.gamma * normalized_x + self.beta
        return self.z
