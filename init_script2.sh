sudo apt update -y
sudo add-apt-repository -y ppa:deadsnakes/ppa 
sudo apt update -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.9
sudo apt-get install -y nano
sudo apt-get install -y python3.9-distutils
yes | poetry install
git clone https://github.com/huggingface/transformers.git
cd transformers
yes | pip install updgrade pip
pip install -e .
cd ..
git config --global user.name "Rana Banerjee"
git config --global user.email "rana1224@gmail.com"
