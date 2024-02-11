""" Implementation of a Fourier-based compressed sensing operator.
"""

from math import sqrt

import torch

from mlsp.model.inpainting import Inpainting
from mlsp.model.linear_operator import LinearOperator

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)


class SensingOperator(LinearOperator):
    r"""Model class implementing a compressed sensing operator based on the
    Fourier transform.

    Attributes
    ----------
    image_size : torch.Size, containing ``d`` elements.
        Full image size.
    mask_id : torch.Tensor of int
        Array of indices corresponding to the observed points.
    weights : torch.Tensor
        Weights for the Fourier measurements.

    Note
    ----
    The implementation follows the description of a Fourier-based random
    sensing operator described in :cite:p:`Candes2008` and :cite:p:`Yang2020`.
    """

    def __init__(
        self,
        image_size: torch.Size,
        mask_id: torch.Tensor,
        weights: torch.Tensor,
    ):
        r"""SensingOperator constructor.

        Parameters
        ----------
        image_size : torch.Size, containing ``d`` elements.
            Full image size.
        mask_id : torch.Tensor of int
            Array of indices corresponding to the observed points.
        weights : torch.Tensor
            Weights for the Fourier measurements.

        Raises
        ------
        # TODO: complete comments
        ValueError
            ``mask`` and image should have the same size.
        """
        # https://pytorch.org/docs/stable/generated/torch.fft.rfft2.html#torch.fft.rfft2
        # ! omit negatives frequencies in the last dimension (no 0-padding)
        self.rfft_size = torch.Size((image_size[0], image_size[1] // 2 + 1))

        if not torch.allclose(
            torch.tensor(weights.shape), torch.tensor(self.rfft_size)
        ):
            raise ValueError("weights and FFT plane should have the same size")

        # ! flattened data dimension after applying the inpainting (= cropping)
        # ! operator
        # ! mask applied in the Fourier domain without zero-padding
        data_size = mask_id.shape
        super(SensingOperator, self).__init__(image_size, data_size)

        self.inpainting_model = Inpainting(self.rfft_size, mask_id)
        self.weights = weights
        self.operator_normalization = sqrt(
            self.image_size[0] * self.image_size[1] / self.data_size[0]
        )
        self.adj_normalization = self.image_size[0] * self.image_size[1]

    def forward(self, input_image):
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space, cropping).

        Parameters
        ----------
        input_image : torch.Tensor of float
            Input image (image space).

        Returns
        -------
        torch.Tensor
            Result of the inpaiting operator (direct operator).
        """
        return self.operator_normalization * self.inpainting_model.forward(
            torch.fft.rfft2(input_image, s=self.image_size)
        )

    def adjoint(self, input_data):
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space, zero-padding).

        Parameters
        ----------
        input_data : torch.Tensor of float
            Input array (2D Fourier space).

        Returns
        -------
        torch.Tensor
            Result of the adjoint of the sensing operator.
        """
        # ! scaling 1/2 needed before irfft2 to have correct adjoint with rfft
        # see for instance https://github.com/PyLops/pylops/issues/268
        y = self.inpainting_model.adjoint(
            self.operator_normalization * self.adj_normalization * input_data
        )
        out = torch.empty_like(y)
        out.copy_(y)
        out[..., 1:-1] /= 2
        return torch.fft.irfft2(out, s=self.image_size)


if __name__ == "__main__":
    from inpainting import generate_random_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = torch.Generator(device=device)
    rng.manual_seed(1234)

    image_size = torch.Size((10, 10))
    percent = 0.6

    fourier_size = torch.Size((image_size[0], image_size[1] // 2 + 1))
    masked_id, mask = generate_random_mask(fourier_size, percent, rng)
    weights = torch.ones(fourier_size)
    cs_operator = SensingOperator(image_size, masked_id, weights)

    print("Number of masked entries: {}".format(masked_id.numel()))

    # * test implementation of the direct and adjoint operators
    x = torch.normal(0.0, 1.0, image_size, generator=rng, device=device)
    y = (1 + 1j) * torch.normal(
        0.0, 1.0, cs_operator.inpainting_model.data_size, generator=rng, device=device
    )

    Ax = cs_operator.forward(x)
    Aadj_y = cs_operator.adjoint(y)

    # identify \mathbb{C} to \mathbb{R}^2 for the computation of the scalar
    # product below (real-valued scalar products needed here for consistency)
    sp1 = torch.sum(torch.real(Ax) * torch.real(y)) + torch.sum(
        torch.imag(Ax) * torch.imag(y)
    )
    sp2 = torch.sum(x * Aadj_y)
    print("Correct adjoint implementation? {0}".format(torch.isclose(sp1, sp2)))

    # test autodiff across cs_operator
    x1 = torch.normal(
        0.0, 1.0, image_size, generator=rng, device=device, requires_grad=True
    )
    loss = torch.norm(cs_operator.forward(x1), p="fro") ** 2 / 2
    print(f"Gradient function for loss = {loss.grad_fn}")
    loss.backward(retain_graph=True)
    # x1.grad.zero_()

    with torch.no_grad():
        reference_gradient = cs_operator.adjoint(cs_operator.forward(x1))

    print(
        "Consistency autodiff. across CS operator? {0}".format(
            torch.allclose(x1.grad, reference_gradient)
        )
    )  # issue in [7, 4] -> manual implementation more precise than autodiff

    rel_error = torch.norm(x1.grad - reference_gradient, p="fro") / torch.norm(
        reference_gradient, p="fro"
    )
    print("Relative error (exact gradient vs autodiff): {0:1.3e}".format(rel_error))
