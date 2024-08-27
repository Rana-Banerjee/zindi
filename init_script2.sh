yes | poetry install
git clone https://github.com/huggingface/transformers.git --branch v4.44.1
cd transformers
yes | pip install updgrade pip
pip install -e .
cd ..
cp common/transformers_bak.py ~/.cache/pypoetry/virtualenvs/zindi-LtLKIbXv-py3.9/lib/python3.9/site-packages/ctranslate2/converters/transformers.py
git config --global user.name "Rana Banerjee"
git config --global user.email "rana1224@gmail.com"
