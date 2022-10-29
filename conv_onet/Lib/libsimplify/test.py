import numpy as np

from conv_onet.Lib.libsimplify.simplify import mesh_simplify

v = np.random.rand(100, 3)
f = np.random.choice(range(100), (50, 3))

mesh_simplify(v, f, 50)
