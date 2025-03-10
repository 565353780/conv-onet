pip install -U cython imageio numpy pandas pillow pyembree pytest pyyaml \
  scikit-image scipy tensorboardx tqdm trimesh h5py plyfile open3d pykdtree

pip install -U torch torchvision torchaudio

#pip install -U torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

python setup.py build_ext --inplace
