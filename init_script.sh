cd zindi
yes | pip install poetry
poetry shell
sudo apt update --force-yes
sudo add-apt-repository ppa:deadsnakes/ppa --force-yes
sudo apt update --force-yes
sudo add-apt-repository ppa:deadsnakes/ppa --force-yes
sudo apt install python3.9 --force-yes
sudo apt-get install nano --force-yes
sudo apt-get install python3.9-distutils --force-yes
yes | poetry install
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ..
git config --global user.name "Rana Banerjee"
git config --global user.email "rana1224@gmail.com"
