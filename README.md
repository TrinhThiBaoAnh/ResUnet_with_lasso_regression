# Installation

```
conda create -n resunet 
conda activate resunet
pip3 install -r requirements.txt
```
# Data preparation

Put your data in this format

```
  train
    | images
    | mask_images
  val
    | images
    | mask_images
  test
    | images
    | mask_images
```

Custom your dataset in dataset.py

# Training

Step 1: Change Model in line 132, 133 in train.py

Step 2: Change data path in line 307 in train.py

Step 3: Run train.py

```
python3 train.py 
```
