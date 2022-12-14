import numpy

def gs(X):
    for w in range(numpy.shape(X)[1]):
        if w == 0:
            X[:, w] = X[:, w]/numpy.linalg.norm(X[:, w])
        else:
            v = X[:, w] - numpy.dot(X[:, w-1], X[:, w]) * X[:, w-1]
            X[:, w] = v/numpy.linalg.norm(v)
    return X

def equipotential_standard_normal(d, n):
    '''Draws n samples from standard normal multivariate gaussian distribution of dimension d which are equipotential
    and are lying on a grand circle (unit d-sphere) on a n-1 manifold which was randomly chosen.
    d: number of dimensions
    n: size of sample
    return: n samples of size d from the standard normal distribution which are equally likely'''
    x = numpy.random.standard_normal((d, 1))  # starting sample

    r = numpy.sqrt(numpy.sum(x ** 2))  # ||x||
    x = x / r  # project sample on d-1-dimensional UNIT sphere --> x just defines direction
    t = numpy.random.standard_normal((d, 1))  # draw tangent sample
    t = t - (numpy.dot(numpy.transpose(t), x) * x)  # Gram Schmidth orthogonalization --> determines which circle is traversed
    t = t / (numpy.sqrt(numpy.sum(t ** 2)))  # standardize ||t|| = 1
    s = numpy.linspace(0, 2 * numpy.pi, n+1)  # space to span --> once around the circle in n steps
    s = s[0:(len(s) - 1)]
    t = s * t #if you wrap this samples around the circle you get once around the circle
    X = r * exp_map(x, t)  # project onto sphere, re-scale
    return (X)


def exp_map(mu, E):
    '''starting from a point mu on the grand circle adding a tangent vector to mu will end at a position outside of the
    circle. Samples need to be maped back on the circle.
    mu: starting sample
    E: tangents of different length from 0 to 2 pi times 1
    returns samples lying onto the unit circle.'''
    D = numpy.shape(E)[0]
    theta = numpy.sqrt(numpy.sum(E ** 2, axis=0))
    M = numpy.dot(mu, numpy.expand_dims(numpy.cos(theta), axis=0)) + E * numpy.sin(theta) / theta
    if (any(numpy.abs(theta) <= 1e-7)):
        for a in (numpy.where(numpy.abs(theta) <= 1e-7)):
            M[:, a] = mu
    M[:, abs(theta) <= 1e-7] = mu
    return (M)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
