import torch

from .basis_utils import get_sph_harm_basis
from .radial_basis import GaussianBasis, RadialBasis
from .scaling import ScaledModule, ScalingFactor


class CircularBasisLayer(ScaledModule):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    radial_basis: RadialBasis
        Radial basis functions
    cbf: dict
        Name and hyperparameters of the cosine basis function
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        cbf: str,
        scale_basis: bool = False,
    ):
        super().__init__()

        self.radial_basis = radial_basis

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_cbf = ScalingFactor()

        cbf_name = cbf["name"].lower()
        cbf_hparams = cbf.copy()
        del cbf_hparams["name"]

        if cbf_name == "gaussian":
            self.cosφ_basis = GaussianBasis(
                start=-1, stop=1, num_gaussians=num_spherical, **cbf_hparams
            )
        elif cbf_name == "spherical_harmonics":
            self.cosφ_basis = get_sph_harm_basis(
                num_spherical, zero_m_only=True
            )
        else:
            raise ValueError(f"Unknown cosine basis function '{cbf_name}'.")

    def forward(self, D_ca, cosφ_cab):
        rbf = self.radial_basis(D_ca)  # (num_edges, num_radial)
        cbf = self.cosφ_basis(cosφ_cab)  # (num_triplets, num_spherical)

        if self.scale_basis:
            cbf = self.scale_cbf(cbf)

        return rbf, cbf
        # (num_edges, num_radial), (num_triplets, num_spherical)


class SphericalBasisLayer(ScaledModule):
    """
    3D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    radial_basis: RadialBasis
        Radial basis functions
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        sbf: str,
        scale_basis: bool = False,
        sin_ϑ: bool = False,
    ):
        super().__init__()

        self.num_spherical = num_spherical
        self.radial_basis = radial_basis

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_sbf = ScalingFactor()

        sbf_name = sbf["name"].lower()
        sbf_hparams = sbf.copy()
        del sbf_hparams["name"]

        assert not sin_ϑ or (sbf_name == "legendre_outer")

        if sbf_name == "spherical_harmonics":
            self.spherical_basis = get_sph_harm_basis(
                num_spherical, zero_m_only=False
            )

        elif sbf_name == "legendre_outer":
            circular_basis = get_sph_harm_basis(
                num_spherical, zero_m_only=True
            )
            if sin_ϑ:
                # The full outer product might be a bit extreme. We should
                # look into more efficient ways of combining cos & sin (concat?).
                self.spherical_basis = lambda cosφ, ϑ: (
                    circular_basis(cosφ)[:, :, None, None]
                    * circular_basis(torch.cos(ϑ))[:, None, :, None]
                    * circular_basis(torch.sin(ϑ))[:, None, None, :]
                ).reshape(cosφ.shape[0], -1)
            else:
                self.spherical_basis = lambda cosφ, ϑ: (
                    circular_basis(cosφ)[:, :, None]
                    * circular_basis(torch.cos(ϑ))[:, None, :]
                ).reshape(cosφ.shape[0], -1)

        elif sbf_name == "gaussian_outer":
            self.circular_basis = GaussianBasis(
                start=-1, stop=1, num_gaussians=num_spherical, **sbf_hparams
            )
            self.spherical_basis = lambda cosφ, ϑ: (
                self.circular_basis(cosφ)[:, :, None]
                * self.circular_basis(torch.cos(ϑ))[:, None, :]
            ).reshape(cosφ.shape[0], -1)

        else:
            raise ValueError(f"Unknown spherical basis function '{sbf_name}'.")

    def forward(self, D_ca, cosφ_cab, θ_cabd):
        rbf = self.radial_basis(D_ca)
        sbf = self.spherical_basis(cosφ_cab, θ_cabd)
        # (num_quadruplets, num_spherical**2)

        if self.scale_basis:
            sbf = self.scale_sbf(sbf)

        return rbf, sbf
        # (num_edges, num_radial), (num_quadruplets, num_spherical**2)
