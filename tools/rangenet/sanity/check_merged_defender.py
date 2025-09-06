import numpy as np

M = np.loadtxt("data/vendor/sph/LIMP_MULTI/25/UTGvBB/oop_defend.csv", delimiter=",")
assert M.shape == (13,13)
assert 0.0 <= M.min() and M.max() <= 1.0
print("ok", M.shape, M.min(), M.max())