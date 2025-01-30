# Reliable and Efficient Amortized Model-based Evaluation
Set up the environment:
```bash
conda create -n reeval python=3.10 -y
conda activate reeval
pip install -r requirements.txt
conda install nvidia/label/cuda-12.1.0::libcurand
conda install nvidia/label/cuda-12.1.0::libcurand-dev
conda env create -f R.yml
```