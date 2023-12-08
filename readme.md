# NAFNet Baslines

## File Structure
This code is built off the [GitHub code](https://github.com/megvii-research/NAFNet) published by NAFNet. These are the files we added or significantly modified for the purpose of testing out our hypothesis.
```
├── basicsr
|   ├── models
|   │   ├── archs
|   │   │   ├── VAE_arch.py
├── options
│   ├── train
│   |   ├── GoPro
│   │   |   ├── VAE-width32.yml
│   ├── test
│   |   ├── GoPro
│   │   |   ├── VAE-width32.yml
```

### Installation
This implementation is based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [HINet](https://github.com/megvii-model/HINet) 

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
git clone https://github.com/macvincent/naafnet_baselines
cd naafnet_baselines
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```
## Running Baseline

### 1. Data Preparation

##### Download the train set and place it in ```./datasets/GoPro/train```:

* [google drive](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing) 
* it should be like ```./datasets/GoPro/train/input ``` and ```./datasets/GoPro/train/target```
* ```python scripts/data_preparation/gopro.py``` to crop the train image pairs to 512x512 patches and make the data into lmdb format.

##### Download the evaluation data (in lmdb format) and place it in ```./datasets/GoPro/test/```:

  * [google drive](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view?usp=sharing)
  * it should be like ```./datasets/GoPro/test/input.lmdb``` and ```./datasets/GoPro/test/target.lmdb```



### 2. Training

* NAFNet Model:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/NAFNet-width32.yml --launcher pytorch
  ```

* VAE with NAFNet Blocks:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/VAE-width32.yml --launcher pytorch
  ```

### 3. Evaluation

After training the model.
  * NAFNet Model:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/GoPro/NAFNet-width32.yml --launcher pytorch
```

  * VAE with NAFNet Blocks:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/GoPro/VAE-width32 --launcher pytorch
```
