""" Implementation of the 2D discrete gradient and its adjoint, with a linear
operator involved when decomposing the operator in the Fourier domain ADMM
splitting (see tutorial 1).
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

import torch

from mlsp.model.linear_operator import LinearOperator

# TODO: complete the functions "gradient_2d" and "gradient_2d_adjoint" at the basis of the implementation of the "DiscreteGradient" class.


def gradient_2d(x: torch.Tensor) -> torch.Tensor:
    r"""Compute 2d discrete gradient.

    Compute the 2d discrete gradient of a 2d input array :math:`\mathbf{x}`,
    **i.e.**, by computing horizontal and vertical differences:

    .. math::
       \nabla(\mathbf{x}) = (\nabla_v\mathbf{x}, \mathbf{x}\nabla_h).

    Parameters
    ----------
    x : torch.Tensor
        Input 2d array :math:`\mathbf{x}`.

    Returns
    -------
    torch.Tensor, of shape ``(2, *x.shape)``
        Vertical and horizontal differences, concatenated along the axis 0.
    """
    assert len(x.shape) == 2, "gradient_2d: Invalid input, expected a 2d tensor"

    # vertical differences
    uv = torch.zeros(x.shape)
    uv[:-1, :] = x[1:, :]
    uv = uv - x
    # horizontal differences
    uh = torch.zeros(x.shape)
    uh[:, :-1] = x[:, 1:]
    uh = uh - x

    return torch.stack((uv, uh), dim=0)


def gradient_2d_adjoint(y: torch.Tensor) -> torch.Tensor:
    r"""Adjoint of the 2d discrete gradient operator.

    Compute the adjoint of the 2d discrete gradient of a 2d input array
    :math:`\mathbf{x}`,

    .. math::
       \nabla^*(\mathbf{y}) = - \text{div} (\mathbf{y})
       = \nabla_v^*\mathbf{y}_v + \mathbf{y}_h\nabla_h^*.

    Parameters
    ----------
    y : torch.Tensor, 3d
        Input array.

    Returns
    -------
    torch.Tensor, of shape ``(y.shape[1], y.shape[2])``
        Adjoint of the 2d gradient operator, evaluated in :math:`\mathbf{y}`.
    """
    # TODO: fill-in computations
    uv, uh = y[0], y[1]
    adj = torch.zeros_like(uv)
    h, w = uv.shape

    for i in range(uv.shape[0] - 1, -1, -1):
        if i == h - 1:
            adj[i] = -uv[i]
        else:
            adj[i] = -uv[i] + adj[i + 1]

    return adj


class DiscreteGradient(LinearOperator):
    """2D discrete gradient linear operator.

    Attributes
    ----------
    image_size : torch.Size
        Size of the input images to which the operator can be applied.
    data_size : torch.Size
        Output size of the 2D discrete gradient operator.
    """

    def __init__(
        self,
        image_size: torch.Size,
    ):
        super(DiscreteGradient, self).__init__(image_size, (2, *image_size))

    forward = staticmethod(gradient_2d)
    adjoint = staticmethod(gradient_2d_adjoint)


if __name__ == "__main__":
    # check consistency implementation of the discrete gradients
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)
    image_size = torch.Size((5, 6))

    x = torch.normal(0.0, 1.0, image_size, generator=rng, device=device)
    y = torch.normal(0.0, 1.0, (2, *image_size), generator=rng, device=device)

    G = DiscreteGradient(image_size)

    Gx = G.forward(x)
    Gstar_y = G.adjoint(y)

    sp1 = torch.sum(Gx * y)
    sp2 = torch.sum(x * Gstar_y)

    print(
        "Correct implementation of the adjoint of the discrete gradient? {0}".format(
            torch.allclose(sp1, sp2)
        )
    )
