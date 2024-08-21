sudo apt update -y
sudo add-apt-repository -y ppa:deadsnakes/ppa 
sudo apt update -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.9
sudo apt-get install -y nano
sudo apt-get install -y python3.9-distutils
yes | pip install poetry
poetry shell
