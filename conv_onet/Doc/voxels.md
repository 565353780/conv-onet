# Voxels

Binvox to Numpy and back.

```bash
>>> import numpy as np
>>> from conv_onet.Data.voxels import Voxels
>>> with open('chair.binvox', 'rb') as f:
...     m1 = Voxels.read_as_3d_array(f)
...
>>> m1.dims
[32, 32, 32]
>>> m1.scale
41.133000000000003
>>> m1.translate
[0.0, 0.0, 0.0]
>>> with open('chair_out.binvox', 'wb') as f:
...     m1.write(f)
...
>>> with open('chair_out.binvox', 'rb') as f:
...     m2 = Voxels.read_as_3d_array(f)
...
>>> m1.dims==m2.dims
True
>>> m1.scale==m2.scale
True
>>> m1.translate==m2.translate
True
>>> np.all(m1.data==m2.data)
True

>>> with open('chair.binvox', 'rb') as f:
...     md = Voxels.read_as_3d_array(f)
...
>>> with open('chair.binvox', 'rb') as f:
...     ms = Voxels.read_as_coord_array(f)
...
>>> data_ds = md.dense_to_sparse()
>>> data_sd = ms.sparse_to_dense(32)
>>> np.all(data_sd==md.data)
True
>>> # the ordering of elements returned by numpy.nonzero changes with axis
>>> # ordering, so to compare for equality we first lexically sort the voxels.
>>> np.all(ms.data[:, np.lexsort(ms.data)] == data_ds[:, np.lexsort(data_ds)])
True
```

