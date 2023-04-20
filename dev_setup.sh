pip install cython imageio numpy pandas pillow pyembree pytest pyyaml \
  scikit-image scipy tensorboardx tqdm trimesh h5py plyfile open3d

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

python setup.py build_ext --inplace

CC=gcc CXX=gcc python cc_setup.py build_ext --inplace
