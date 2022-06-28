from typing import Optional

import torch
from torch_scatter import scatter

from ..initializers import he_orthogonal_init
from .base_layers import Dense


class BasisEmbedding(torch.nn.Module):
    """
    Embed a basis (CBF, SBF), optionally using the efficient reformulation.
    """

    def __init__(
        self,
        num_radial: int,
        emb_size_interm: int,
        num_spherical: Optional[int] = None,
        rbf_per_sph: bool = False,
    ):
        super().__init__()
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.rbf_per_sph = rbf_per_sph
        if num_spherical is None:
            assert not rbf_per_sph
            self.weight = torch.nn.Parameter(
                torch.empty(emb_size_interm, num_radial),
                requires_grad=True,
            )
        else:
            if rbf_per_sph:
                self.weight = torch.nn.Parameter(
                    torch.empty(num_spherical, num_radial, emb_size_interm),
                    requires_grad=True,
                )
            else:
                self.weight = torch.nn.Parameter(
                    torch.empty(num_radial, num_spherical, emb_size_interm),
                    requires_grad=True,
                )
        self.reset_parameters()

    def reset_parameters(self):
        he_orthogonal_init(self.weight)

    def forward(
        self,
        rbf,
        sph=None,
        idx_rad=None,
        idx_ragged_rad=None,
        idx_sph=None,
        idx_ragged_sph=None,
        num_atoms=None,
    ):
        """

        Arguments
        ---------
        rbf: torch.Tensor, shape=(num_edges, num_radial or num_orders * num_radial)
        sph: torch.Tensor, shape=(num_triplets or num_quadruplets, num_spherical)
        id_ca
        id_ragged_idx

        Returns
        -------
        rbf_W1: torch.Tensor, shape=(num_edges, emb_size_interm, num_spherical)
        sph: torch.Tensor, shape=(num_edges, Kmax, num_spherical)
            Kmax = maximum number of neighbors of the edges
        """
        num_edges = rbf.shape[0]

        if self.num_spherical is not None:
            if self.rbf_per_sph:
                rbf = rbf.view(
                    (num_edges, -1, self.num_radial)
                )
                # (num_edges, num_orders, num_radial)
                rbf = torch.transpose(rbf, 0, 1)
                # (num_orders, num_edges, num_radial)
                # sph might still be larger (e.g. num_orders**2)
                rbf = torch.repeat_interleave(
                    rbf, self.num_spherical // rbf.shape[0], dim=0
                )
                # (num_spherical, num_edges, num_radial)
                # MatMul: mul + sum over num_radial
                rbf_W1 = rbf @ self.weight
                # (num_spherical, num_edges, emb_size_interm)
                rbf_W1 = rbf_W1.permute(1, 2, 0)
                # (num_edges, emb_size_interm, num_spherical)
            else:
                # MatMul: mul + sum over num_radial
                rbf_W1 = rbf @ self.weight.reshape(self.weight.shape[0], -1)
                # (num_edges, emb_size_interm * num_spherical)
                rbf_W1 = rbf_W1.reshape(num_edges, -1, sph.shape[-1])
                # (num_edges, emb_size_interm, num_spherical)
        else:
            # MatMul: mul + sum over num_radial
            rbf_W1 = rbf @ self.weight.T
            # (num_edges, emb_size_interm)

        if idx_ragged_rad is not None:
            # Zero padded dense matrix
            # maximum number of neighbors
            if idx_rad.shape[0] == 0:
                # catch empty idx_rad
                Kmax = 0
            else:
                Kmax = torch.max(idx_ragged_rad) + 1

            rbf_W1_padded = rbf_W1.new_zeros(
                [num_atoms, Kmax] + list(rbf_W1.shape[1:])
            )
            rbf_W1_padded[idx_rad, idx_ragged_rad] = rbf_W1
            # (num_atoms, Kmax, emb_size_interm, ...)
            rbf_W1_padded = torch.transpose(rbf_W1_padded, 1, 2)
            # (num_atoms, emb_size_interm, Kmax, ...)
            rbf_W1_padded = rbf_W1_padded.reshape(
                num_atoms, rbf_W1.shape[1], -1
            )
            # (num_atoms, emb_size_interm, Kmax2 * ...)
            rbf_W1 = rbf_W1_padded

        if idx_ragged_sph is not None:
            # Zero padded dense matrix
            # maximum number of neighbors
            if idx_sph.shape[0] == 0:
                # catch empty idx_sph
                Kmax = 0
            else:
                Kmax = torch.max(idx_ragged_sph) + 1

            sph2 = sph.new_zeros(num_edges, Kmax, sph.shape[-1])
            sph2[idx_sph, idx_ragged_sph] = sph
            # (num_edges, Kmax, num_spherical)
            sph2 = torch.transpose(sph2, 1, 2)
            # (num_edges, num_spherical, Kmax)

        if sph is None:
            return rbf_W1
        else:
            if idx_ragged_sph is None:
                rbf_W1 = rbf_W1[idx_sph]
                # (num_triplets, emb_size_interm, num_spherical)

                cbf_W1 = rbf_W1 @ sph[:, :, None]
                # (num_triplets, emb_size_interm, num_spherical)
                return cbf_W1.squeeze(-1)
            else:
                return rbf_W1, sph2


class EfficientInteractionBilinear(torch.nn.Module):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.
    """

    def __init__(
        self,
        emb_size_in: int,
        emb_size_interm: int,
        emb_size_out: int,
    ):
        super().__init__()
        self.emb_size_in = emb_size_in
        self.emb_size_interm = emb_size_interm
        self.emb_size_out = emb_size_out

        self.bilinear = Dense(
            self.emb_size_in * self.emb_size_interm,
            self.emb_size_out,
            bias=False,
            activation=None,
        )

    def forward(
        self,
        basis,
        m,
        id_reduce,
        id_ragged_idx,
        id_reduce2=None,
        id_ragged_idx2=None,
        reduce_size2=None,
    ):
        """

        Arguments
        ---------
        basis
        m: quadruplets: m = m_db , triplets: m = m_ba
        id_reduce
        id_ragged_idx

        Returns
        -------
            m_ca: torch.Tensor, shape=(num_edges, emb_size)
                Edge embeddings.
        """
        # num_spherical is actually num_spherical**2 for quadruplets
        (rbf_W1, sph) = basis
        # (num_edges, emb_size_interm, num_spherical),
        # (num_edges, num_spherical, Kmax)
        num_edges = sph.shape[0]

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        Kmax = torch.max(id_ragged_idx) + 1
        m_padded = m.new_zeros(num_edges, Kmax, self.emb_size_in)
        m_padded[id_reduce, id_ragged_idx] = m
        # (num_quadruplets/num_triplets, emb_size_in) -> (num_edges, Kmax, emb_size_in)

        sph_m = torch.matmul(sph, m_padded)
        # (num_edges, num_spherical, emb_size_in)

        if id_reduce2 is not None:
            Kmax2 = torch.max(id_ragged_idx2) + 1
            sph_m_padded = sph_m.new_zeros(
                reduce_size2, Kmax2, sph_m.shape[1], sph_m.shape[2]
            )
            sph_m_padded[id_reduce2, id_ragged_idx2] = sph_m
            # (num_atoms, Kmax2, num_spherical, emb_size_in)
            sph_m_padded = sph_m_padded.reshape(
                reduce_size2, -1, sph_m.shape[-1]
            )
            # (num_atoms, Kmax2 * num_spherical, emb_size_in)

            rbf_W1_sph_m = rbf_W1 @ sph_m_padded
            # (num_atoms, emb_size_interm, emb_size_in)
        else:
            # MatMul: mul + sum over num_spherical
            rbf_W1_sph_m = torch.matmul(rbf_W1, sph_m)
            # (num_edges, emb_size_interm, emb_size_in)

        # Bilinear: Sum over emb_size_interm and emb_size_in
        m_ca = self.bilinear(
            rbf_W1_sph_m.reshape(-1, rbf_W1_sph_m.shape[1:].numel())
        )
        # (num_edges/num_atoms, emb_size_out)

        return m_ca
