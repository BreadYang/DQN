cd /pylon2/$(id -gn)/$USER

module load python3
virtualenv deeprl-hw2-gpu
source deeprl-hw2-gpu/bin/activate
#pip install tensorflow-gpu
pip install gym
pip install -U numpy
pip install attrs
pip install h5py
pip install keras
pip install matplotlib
pip install pillow
pip install protobuf>=3.0
pip install pydot-ng
pip install scipy
pip install semver
pip install tensorboard
pip install atari-py
pip install tensorflow-gpu

deactivate
