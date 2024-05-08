pip install -U cython imageio numpy pandas pillow pyembree pytest pyyaml \
	scikit-image scipy tensorboardx tqdm trimesh h5py plyfile open3d

pip install -U torch torchvision torchaudio

pip install -U torch-scatter

python setup.py build_ext --inplace

CC=gcc CXX=gcc python cc_setup.py build_ext --inplace
