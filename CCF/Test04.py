# import scipy.stats as stats
# import matplotlib.pyplot as plt
# import numpy as np
# a = np.arange(16)
# poi = stats.poisson
# lambda_ = [1.5, 4.25]
# colours = ["#348ABD", "#A60628"]
# plt.bar(a, poi.pmf(a, lambda_[0]), color=colours[0],label="$\lambda = %.1f$" % lambda_[0], alpha=0.60,edgecolor=colours[0], lw="3")
# plt.bar(a, poi.pmf(a, lambda_[1]), color=colours[1],label="$\lambda = %.1f$" % lambda_[1], alpha=0.60,edgecolor=colours[1],lw="3")
# plt.xticks(a + 0.4, a)
# plt.legend()
# plt.ylabel("probability of $k$")
# plt.xlabel("$k$")
# plt.title("Probability mass function of a Poisson random variable; differing \$\lambda$ values")
# plt.show()
import pymc as pm
parameter = pm.Exponential("poisson_param", 1)
data_generator = pm.Poisson("data_generator", parameter)
data_plus_one = data_generator + 1