pip install cython imageio numpy matplotlib pandas pillow pyembree pytest pyyaml \
  scikit-image scipy tensorboardx tqdm trimesh h5py plyfile

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

python setup.py build_ext --inplace

