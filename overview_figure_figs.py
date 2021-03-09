import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == '__main__':
    x = np.random.standard_normal(size=(10, 5))
    f = plt.figure()
    plt.imshow(x)
    plt.axis('off')
    plt.savefig('matrix.png', bbox_inches='tight')

    f = plt.figure()
    ax1 = f.add_subplot(111)
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax1.spines['left'].set_position(('data', 0))
    ax1.spines['bottom'].set_position(('data', 0))

    # Eliminate upper and right axes
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')

    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])

    ax1.set_xticks([])
    ax1.set_yticks([])

    mu = 0
    variance = 1
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), c='green')

    plt.savefig('gaussian.png', bbox_inches='tight')