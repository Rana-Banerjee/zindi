wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
sudo apt-get install -y nvidia-open
sudo apt-get install libprotobuf10 protobuf-compiler libprotobuf-dev
rm -rf cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb

git clone https://github.com/marian-nmt/marian
cd marian
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_SENTENCEPIECE=ON
make -j 8

git clone https://github.com/marian-nmt/sacreBLEU.git sacreBLEU