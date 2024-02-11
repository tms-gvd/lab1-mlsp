# MLSP labs, M2DS: setup and general instructions

Author: P.-A. Thouvenin

- All the files in this lab are distributed under [GPL-3.0 license](LICENSE.md).
- Corresponding research papers and sources (including this lecture) need to be cite in case some of the codes are reused.

## Create a `conda` environment for the labs

- If you are working on your own machine, we strongly recommand using [`mamba`](https://mamba.readthedocs.io/en/latest/), a fast C++ reimplementation of [`conda`](https://docs.anaconda.com/free/anaconda/install/), with the same command line interface and access to the same `conda`-channels.

- Create a lab environment using one of the provided `.yml` file (use the file corresponding to your operating system (OS))

```bash
# for linux / WSL
mamba env create --name mlsp --file mlsp_linux.lock.yml

# for MacOS
# mamba env create --name mlsp --file mlsp_osx.lock.yml

mamba activate mlsp

# install mlsp library in dev mode
conda develop src
```

- In case the provided `.yml` files do not work on your system, you can manually install the required packages as follows.

```bash
mamba env create --name mlsp pytorch::pytorch torchvision torchaudio -c pytorch -y
mamba activate mlsp
pip install deepinv
mamba install tqdm jupyterlab pytest black flake8 isort pre-commit conda-build -y
mamba install scikit-image h5py -y

# install mlsp library in development mode
conda develop src
```

## Retrieve pre-trained networks for PnP regularization

The pretrained networks used in this work as PnP priors can be directly retrieved from [github](https://github.com/cszn/KAIR) as follows (assuming `ubuntu`, `MacOS` or [`WSL`](https://learn.microsoft.com/en-us/windows/wsl/install)).

```bash
mkdir labs/weights

# DnCNN
wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth

# FFDNet
wget https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray.pth

# DRUNet
wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth

# SCUNet
wget https://github.com/cszn/KAIR/releases/download/v1.0/scunet_gray_15.pth
```

The codes retrieved from this [github repository](https://github.com/cszn/KAIR) have been made available by their author under MIT license.

Example usage are provided in the lab guidelines to indicate how to apply these networks to an input image.

## Lab report

- Lab reports are to be prepared in pairs. Groups of more than 2 students are not allowed.

- Rename your lab archive report with a name of the form `mlsp_labx_Name1_Name2`, with `x` replaced by the lab number, and `Name1`, `Name2` replaced by the name of the students working on the report.

- You report will have to be uploaded on Moodle by the prescribed deadline. It should contain ONLY your jupyter notebook `.ipynb`, saved results and any Python file you have used to produce the results. `img/` and `weights/` folder for the deep priors need to be removed from the archive.
