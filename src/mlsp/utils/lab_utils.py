from math import sqrt
from os.path import splitext

import h5py
import numpy as np
import torch
from PIL import Image

from mlsp.model.cs_operator import SensingOperator
from mlsp.model.inpainting import Inpainting, generate_random_mask

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# Part of the code from this lab is directly adapted from the Python code
# available https://gitlab.cristal.univ-lille.fr/pthouven/ownsamplinggs (GPL 3.0
# license), associated with the folllowing paper
#
# P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

# TODO: complete the function "generate_2d_cs_data"


@torch.no_grad
def load_and_normalize_image(
    imagefilename: str,
    device,
    max_intensity: float | None = None,
    downsampling: int | None = None,
):
    r"""Load an image from a .png of .h5 file, ensuring all pixel values are
    nonnegative.

    Parameters
    ----------
    imagefilename : str
        Full path and name of the image to be loaded (including extension).
    device: torch.cuda.device
        Device on which the returned torch array will be defined (CPU or GPU).
    max_intensity : double, optional
        Maximum intensity value selected, by default None.
    downsampling : int or None, optional
        Downsampling factor to load a smaller image.

    Returns
    -------
    torch.ndarray
        Loaded image, of shape (1, H, W) (1 channel considered).
    """
    file_extension = splitext(imagefilename)[-1]

    if file_extension == ".h5":
        with h5py.File(imagefilename, "r") as f:
            x = torch.tensor(f["x"][()], dtype=torch.float32)
    else:  # .png file by default
        img = np.asarray(Image.open(imagefilename, "r"))
        if downsampling is not None:
            img = img[0:-1:downsampling, 0:-1:downsampling]
        x = torch.as_tensor(img.copy(), dtype=torch.float32, device=device)

    # make sure no pixel is 0 or lower, and renormalize image in [0, M]
    if max_intensity is None:
        max_intensity = torch.max(x)
    x[x <= 0] = torch.min(x[x > 0])  # np.finfo(x.dtype).eps
    x = x * (max_intensity / torch.max(x))

    return x


@torch.no_grad
def generate_2d_cs_data(x: torch.Tensor, percent: float, isnr: float, rng):
    r"""Generate compressed sensing data obtained from sparse measurements
    in the 2D Fourier domain.

    Parameters
    ----------
    x : torch.Tensor
        Ground truth image.
    percent : float
        Fraction of observed pixels in the Fourier domain, ``1-p``
        corresponding to the fraction of masked pixels in the image (``p``
        needs to be nonnegative, smaller than or equal to 1).
    isnr : float
        Input SNR in dB.
    rng : torch._C.Generator
        Torch random number generator.

    Returns
    -------
    observations : torch.Tensor
        Noisy observations in :math:`\mathbb{C}^M`.
    sensing_operator : SensingOperator
        Fourier-based compressed sensing operator.
    sig : torch.Tensor
        Standard deviation of the Gaussian noise.
    clean_observations : torch.Tensor
        Noise-free observations.
    """
    # * generate a synthetic Fourier-based CS operator
    image_size = x.shape
    fourier_size = torch.Size((*image_size[:-1], image_size[-1] // 2 + 1))
    masked_id, mask = generate_random_mask(fourier_size, percent, rng)
    weights = torch.ones(fourier_size)
    sensing_operator = SensingOperator(image_size, masked_id, weights)
    n_measurements = sensing_operator.data_size[0]

    # * generate observations
    # TODO: to complete definition of sig, the standard deviation of the noise
    clean_observations = torch.zeros(n_measurements)
    sig = torch.ones((1,))
    observations = torch.zeros(n_measurements)

    return observations, sensing_operator, sig, clean_observations


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = load_and_normalize_image(
        "labs/img/barb.png", device, max_intensity=None, downsampling=4
    )
    print("Image size: {}".format(x.shape))

    isnr = 30
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)

    percent = 0.6
    (
        observations,
        sensing_operator,
        sig,
        radius,
        clean_observations,
    ) = generate_2d_cs_data(x, percent, isnr, rng)

    print("Image size: {}".format(x.shape))
    print("Number of observations: {}".format(sensing_operator.data_size))
