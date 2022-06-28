import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_max_neighbors_mask,
    get_pbc_distances,
    radius_graph_pbc,
)
from torch_geometric.nn import radius_graph
from torch_scatter import scatter, segment_coo
from torch_sparse import SparseTensor

from gemnet_oc.gemnet_q.model.initializers import get_initializer
from gemnet_oc.gemnet_q.model.layers.atom_update_block import OutputBlock
from gemnet_oc.gemnet_q.model.layers.base_layers import Dense, ResidualLayer
from gemnet_oc.gemnet_q.model.layers.efficient import BasisEmbedding
from gemnet_oc.gemnet_q.model.layers.embedding_block import AtomEmbedding, EdgeEmbedding
from gemnet_oc.gemnet_q.model.layers.force_scaler import ForceScaler
from gemnet_oc.gemnet_q.model.layers.interaction_block import InteractionBlock
from gemnet_oc.gemnet_q.model.layers.radial_basis import RadialBasis
from gemnet_oc.gemnet_q.model.layers.scaling import ScaledModule
from gemnet_oc.gemnet_q.model.layers.spherical_basis import CircularBasisLayer, SphericalBasisLayer
from gemnet_oc.gemnet_q.model.utils import (
    get_angle,
    get_edge_id,
    get_ragged_idx,
    inner_product_clamped,
    mask_neighbors,
    masked_select_sparsetensor_flat,
    repeat_blocks,
)


@registry.register_model("latent_gemnet_dev")
class GemNet(ScaledModule):
    """
    Parameters
    ----------
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip_in: int
            (Down-projected) embedding size of the quadruplet edge embeddings before the bilinear layer.
        emb_size_trip_out: int
            (Down-projected) embedding size of the quadruplet edge embeddings after the bilinear layer.
        emb_size_quad_in: int
            (Down-projected) embedding size of the quadruplet edge embeddings before the bilinear layer.
        emb_size_quad_out: int
            (Down-projected) embedding size of the quadruplet edge embeddings after the bilinear layer.
        emb_size_aint_in: int
            Embedding size in the atom interaction before the bilinear layer.
        emb_size_aint_out: int
            Embedding size in the atom interaction after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).

        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        regress_forces: bool
            Whether to predict forces. Default: True
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.

        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        cutoff_qint: float
            Quadruplet interaction cutoff for interatomic directions in Angstrom. No effect for GemNet-(d)T
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
        forces_coupled: bool
            No effect if direct_forces is False. If True enforce that |F_st| = |F_ts|
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the pytorch file containing the scaling factors.
        quad_interaction: bool
            Whether to use quadruplet interactions (with dihedral angles)
    """

    def __init__(
        self,
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip_in: int,
        emb_size_trip_out: int,
        emb_size_quad_in: int,
        emb_size_quad_out: int,
        emb_size_aint_in: int,
        emb_size_aint_out: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        num_output_afteratom: int,
        regress_forces: bool = True,
        direct_forces: bool = False,
        use_pbc: bool = True,
        scale_backprop_forces: bool = False,
        cutoff: float = 6.0,
        cutoff_qint: Optional[float] = None,
        cutoff_aeaint: Optional[float] = None,
        cutoff_aint: Optional[float] = None,
        max_neighbors: int = 50,
        max_neighbors_qint: Optional[int] = None,
        max_neighbors_aeaint: Optional[int] = None,
        max_neighbors_aint: Optional[int] = None,
        rbf: dict = {"name": "gaussian"},
        rbf_spherical: Optional[dict] = None,
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        sbf: dict = {"name": "spherical_harmonics"},
        extensive: bool = True,
        forces_coupled: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        scale_file: Optional[str] = None,
        num_atom_emb_layers: int = 0,
        num_global_out_layers: int = -1,
        quad_interaction: bool = False,
        atom_edge_interaction: bool = False,
        edge_atom_interaction: bool = False,
        atom_interaction: bool = False,
        scale_basis: bool = False,
        qint_tags: list = [0, 1, 2],
        symmetric_edge_symmetrization: bool = False,
        symmetric_mp: bool = True,
        use_parity: bool = False,
        **kwargs,  # backwards compatibility with deprecated arguments
    ):
        super().__init__()
        if len(kwargs) > 0:
            logging.warning(f"Unrecognized arguments: {list(kwargs.keys())}")
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.extensive = extensive

        self.atom_edge_interaction = atom_edge_interaction
        self.edge_atom_interaction = edge_atom_interaction
        self.atom_interaction = atom_interaction
        self.quad_interaction = quad_interaction
        self.qint_tags = torch.tensor(qint_tags)
        self.symmetric_edge_symmetrization = symmetric_edge_symmetrization
        self.use_parity = use_parity
        if not rbf_spherical:
            rbf_spherical = rbf
        rbf_per_sph = (rbf_spherical["name"] == "multi_order_spherical_bessel")

        self.cutoff = cutoff
        self.use_pbc = use_pbc

        if (
            not (self.atom_edge_interaction or self.edge_atom_interaction)
            or cutoff_aeaint is None
        ):
            self.cutoff_aeaint = self.cutoff
        else:
            self.cutoff_aeaint = cutoff_aeaint
        if not self.quad_interaction or cutoff_qint is None:
            self.cutoff_qint = self.cutoff
        else:
            self.cutoff_qint = cutoff_qint
        if not self.atom_interaction or cutoff_aint is None:
            self.cutoff_aint = max(
                self.cutoff,
                self.cutoff_aeaint,
                self.cutoff_qint,
            )
        else:
            self.cutoff_aint = cutoff_aint

        assert self.cutoff <= self.cutoff_aint
        assert self.cutoff_aeaint <= self.cutoff_aint
        assert self.cutoff_qint <= self.cutoff_aint

        self.max_neighbors = max_neighbors

        if (
            not (self.atom_edge_interaction or self.edge_atom_interaction)
            or max_neighbors_aeaint is None
        ):
            self.max_neighbors_aeaint = self.max_neighbors
        else:
            self.max_neighbors_aeaint = max_neighbors_aeaint
        if not self.quad_interaction or max_neighbors_qint is None:
            self.max_neighbors_qint = self.max_neighbors
        else:
            self.max_neighbors_qint = max_neighbors_qint
        if not self.atom_interaction or max_neighbors_aint is None:
            self.max_neighbors_aint = max(
                self.max_neighbors,
                self.max_neighbors_aeaint,
                self.max_neighbors_qint,
            )
        else:
            self.max_neighbors_aint = max_neighbors_aint

        assert self.max_neighbors <= self.max_neighbors_aint
        assert self.max_neighbors_aeaint <= self.max_neighbors_aint
        assert self.max_neighbors_qint <= self.max_neighbors_aint

        self.forces_coupled = forces_coupled

        self.regress_forces = regress_forces

        # GemNet variants
        self.direct_forces = direct_forces
        self.force_scaler = ForceScaler(enabled=scale_backprop_forces)

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        )
        radial_basis_spherical = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf_spherical,
            envelope=envelope,
            scale_basis=scale_basis,
        )
        if self.quad_interaction:
            radial_basis_spherical_qint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_qint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_qint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical_qint,
                cbf=cbf,
                scale_basis=scale_basis,
            )

            self.sbf_basis_qint = SphericalBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical,
                sbf=sbf,
                scale_basis=scale_basis,
                sin_ϑ=self.use_parity,
            )
        if self.atom_edge_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_aeint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical,
                cbf=cbf,
                scale_basis=scale_basis,
            )
        if self.edge_atom_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            radial_basis_spherical_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_eaint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical_aeaint,
                cbf=cbf,
                scale_basis=scale_basis,
            )
        if self.atom_interaction:
            self.radial_basis_aint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )

        self.cbf_basis_tint = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_spherical,
            cbf=cbf,
            scale_basis=scale_basis,
        )
        ### ------------------------------------------------------------------------------------- ###

        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        if self.quad_interaction:
            self.mlp_rbf_qint = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_qint = BasisEmbedding(
                num_radial, emb_size_cbf, num_spherical, rbf_per_sph)
            if self.use_parity:
                self.mlp_sbf_qint = BasisEmbedding(
                    num_radial, emb_size_sbf, num_spherical ** 3, rbf_per_sph)
            else:
                self.mlp_sbf_qint = BasisEmbedding(
                    num_radial, emb_size_sbf, num_spherical ** 2, rbf_per_sph)

        if self.atom_edge_interaction:
            self.mlp_rbf_aeint = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_aeint = BasisEmbedding(
                num_radial, emb_size_cbf, num_spherical, rbf_per_sph)
        if self.edge_atom_interaction:
            self.mlp_rbf_eaint = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_eaint = BasisEmbedding(
                num_radial, emb_size_cbf, num_spherical, rbf_per_sph)
        if self.atom_interaction:
            self.mlp_rbf_aint = BasisEmbedding(num_radial, emb_size_rbf)

        self.mlp_rbf_tint = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf_tint = BasisEmbedding(
            num_radial, emb_size_cbf, num_spherical, rbf_per_sph)

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        # Interaction Blocks
        int_blocks = []
        for _ in range(num_blocks):
            int_blocks.append(
                InteractionBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip_in=emb_size_trip_in,
                    emb_size_trip_out=emb_size_trip_out,
                    emb_size_quad_in=emb_size_quad_in,
                    emb_size_quad_out=emb_size_quad_out,
                    emb_size_aint_in=emb_size_aint_in,
                    emb_size_aint_out=emb_size_aint_out,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    num_atom_emb_layers=num_atom_emb_layers,
                    quad_interaction=quad_interaction,
                    atom_edge_interaction=atom_edge_interaction,
                    edge_atom_interaction=edge_atom_interaction,
                    atom_interaction=atom_interaction,
                    activation=activation,
                    symmetric_mp=symmetric_mp,
                )
            )
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        out_blocks = []
        for _ in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=direct_forces,
                    output_emb=num_global_out_layers >= 0,
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)

        if num_global_out_layers >= 0:
            out_mlp_E = [
                Dense(
                    emb_size_atom * (num_blocks + 1),
                    emb_size_atom,
                    activation=activation,
                )
            ]
            out_mlp_E += [
                ResidualLayer(
                    emb_size_atom,
                    activation=activation,
                )
                for _ in range(num_global_out_layers)
            ]
            self.out_mlp_E = torch.nn.Sequential(*out_mlp_E)
            self.out_energy = Dense(
                emb_size_atom, num_targets, bias=False, activation=None
            )
            if direct_forces:
                out_mlp_F = [
                    Dense(
                        emb_size_edge * (num_blocks + 1),
                        emb_size_edge,
                        activation=activation,
                    )
                ]
                out_mlp_F += [
                    ResidualLayer(
                        emb_size_edge,
                        activation=activation,
                    )
                    for _ in range(num_global_out_layers)
                ]
                self.out_mlp_F = torch.nn.Sequential(*out_mlp_F)
                self.out_forces = Dense(
                    emb_size_edge, num_targets, bias=False, activation=None
                )

            initializer = get_initializer(output_init)
            self.out_energy.reset_parameters(initializer)
            if direct_forces:
                self.out_forces.reset_parameters(initializer)
        else:
            self.out_mlp_E = None
            self.out_mlp_F = None

        # Set shared parameters for better gradients
        self.shared_parameters = [
            (self.mlp_rbf_tint.linear.weight, self.num_blocks),
            (self.mlp_cbf_tint.weight, self.num_blocks),
            (self.mlp_rbf_h.linear.weight, self.num_blocks),
            (self.mlp_rbf_out.linear.weight, self.num_blocks + 1),
        ]
        if self.quad_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_qint.linear.weight, self.num_blocks),
                (self.mlp_cbf_qint.weight, self.num_blocks),
                (self.mlp_sbf_qint.weight, self.num_blocks),
            ]
        if self.atom_edge_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_aeint.linear.weight, self.num_blocks),
                (self.mlp_cbf_aeint.weight, self.num_blocks),
            ]
        if self.edge_atom_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_eaint.linear.weight, self.num_blocks),
                (self.mlp_cbf_eaint.weight, self.num_blocks),
            ]
        if self.atom_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_aint.weight, self.num_blocks),
            ]

        # Load scaling factors
        if scale_file is not None:
            if os.path.isfile(scale_file):
                scales = torch.load(scale_file, map_location="cpu")
                self.load_scales(scales)
            else:
                logging.warning(
                    f"Scale file '{scale_file}' does not exist. "
                    f"The model will use unit scaling factors "
                    f"unless it was loaded from a checkpoint."
                )

    def calculate_quad_angles(
        self,
        V_st,
        V_qint_st,
        id3_db,
        id3_qint_ba_abd,
        id3_ca,
        id3_qint_ba_cab,
        id3_to_id4_abd,
        id3_to_id4_cab,
    ):
        """Calculate angles for quadruplet-based message passing.

        Parameters
        ----------
            V_st: Tensor, shape = (nAtoms, 3)
                Normalized directions from s to t
            id3_ca: torch.Tensor, shape (num_triplets,)
                Indices of output edge c->a of each triplet b->a<-c
            id3_ba: torch.Tensor, shape (num_triplets,)
                Indices of input edge b->a of each triplet b->a<-c
            id3_ba_out: torch.Tensor, shape (num_triplets_qint,)
                Indices of output edge in triplet d->b->a.
            id3_db: torch.Tensor, shape (num_triplets_qint,)
                Indices of input edge in triplet d->b->a.
            id3_to_id4_abd: torch.Tensor, shape (num_quadruplets,)
                Indices to map from triplet d->b->a to quadruplet d->b->a<-c.
            id3_to_id4_cab: torch.Tensor, shape (num_quadruplets,)
                Indices to map from triplet c->a<-b to quadruplet d->b->a<-c.

        Returns
        -------
            cosφ_cab: Tensor, shape = (num_triplets_inint,)
                Cosine of angle between atoms c -> a <- b.
            cosφ_abd: Tensor, shape = (num_triplets_qint,)
                Cosine of angle between atoms a -> b -> d.
            angle_cabd: Tensor, shape = (num_quadruplets,)
                Dihedral angle between atoms c <- a-b -> d.
        """
        # ---------------------------------- d -> b -> a ---------------------------------- #
        V_ba = V_qint_st[id3_qint_ba_abd]  # (num_triplets_qint, 3)
        V_db = V_st[id3_db]  # (num_triplets_qint, 3)
        cosφ_abd = inner_product_clamped(V_ba, V_db)
        # (num_triplets_qint,)

        # Project for calculating dihedral angle
        # Cross product is the same as projection, just 90° rotated
        V_db_cross = torch.cross(V_db, V_ba, dim=-1)  # a - b -| d
        V_db_cross = V_db_cross[id3_to_id4_abd]  # (num_quadruplets,)

        # --------------------------------- c -> a <- b ---------------------------------- #
        V_ca = V_st[id3_ca]  # (num_triplets_in, 3)
        V_ba = V_qint_st[id3_qint_ba_cab]  # (num_triplets_in, 3)
        cosφ_cab = inner_product_clamped(V_ca, V_ba)  # (n4Triplets,)

        # Project for calculating dihedral angle
        # Cross product is the same as projection, just 90° rotated
        V_ca_cross = torch.cross(V_ca, V_ba, dim=-1)  # c |- a - b
        V_ca_cross = V_ca_cross[id3_to_id4_cab]  # (num_quadruplets,)

        # -------------------------------- c -> a - b <- d -------------------------------- #
        half_angle_cabd = get_angle(V_ca_cross, V_db_cross)
        # (num_quadruplets,)
        if self.use_parity:
            # ||V_ca_cross|| * ||V_db_cross|| * sin(angle_cabd) * V_ba,
            # i.e. this cross product points along or against V_ba
            cross_cross = torch.cross(V_ca_cross, V_db_cross, dim=-1)
            cross_cross_norm = cross_cross.norm(dim=-1)
            # This will either be +1 or -1, depending on sign(sin(angle_cabd))
            parity = inner_product_clamped(
                cross_cross / cross_cross_norm.view(-1, 1), V_ba[id3_to_id4_cab]
            )
            # Fix parity when cross_cross is 0 (at 0° and 180°)
            parity[
                torch.isclose(
                    cross_cross_norm, torch.tensor([0.0], device=cross_cross.device)
                )
            ] = 1
            angle_cabd = parity * half_angle_cabd
        else:
            angle_cabd = half_angle_cabd

        return cosφ_cab, cosφ_abd, angle_cabd

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct
        (i.e. stem from different unit cells).

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)
        num_edges = idx_s.size(0)

        value = torch.arange(num_edges, device=idx_s.device, dtype=idx_s.dtype)
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # id3_ca has to be sorted for this
        id3_ragged_idx = get_ragged_idx(id3_ca, dim_size=num_edges)

        return id3_ba, id3_ca, id3_ragged_idx

    def get_mixed_triplets(
        self,
        edge_index_in,
        edge_index_out,
        cell_offsets_in,
        cell_offsets_out,
        num_atoms,
        to_outedge=False,
        return_adj=False,
        return_ragged=False,
    ):
        """
        Get all output edges (ingoing or outgoing) for each incoming edge.
        It is possible that in atom=out atom, as long as the edges are distinct
        (i.e. they stem from different unit cells).

        Returns
        -------
        id3_in: torch.Tensor, shape (num_triplets,)
            Indices of input edges
        id3_out: torch.Tensor, shape (num_triplets,)
            Indices of output edges
        adj_edges: SparseTensor, shape (num_edges, num_atoms)
            Adjacency (incidence) matrix between output edges and atoms,
            with values specifying the input edges
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_out_s, idx_out_t = edge_index_out  # c->a (source=c, target=a)
        idx_in_s, idx_in_t = edge_index_in
        num_edges = idx_out_s.size(0)

        value_in = torch.arange(
            idx_in_s.size(0), device=idx_in_s.device, dtype=idx_in_s.dtype
        )
        # This exploits that SparseTensor can have multiple copies of the same edge!
        adj_in = SparseTensor(
            row=idx_in_t,
            col=idx_in_s,
            value=value_in,
            sparse_sizes=(num_atoms, num_atoms),
        )
        if to_outedge:
            adj_edges = adj_in[idx_out_s]
        else:
            adj_edges = adj_in[idx_out_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_in = adj_edges.storage.value()
        id3_out = adj_edges.storage.row()

        # Remove self-loop triplets c->a<-c or c<-a<-c
        # Check atom as well as cell offset
        if to_outedge:
            id3_atom_in = idx_in_s[id3_in]
            id3_atom_out = idx_out_t[id3_out]
            cell_offsets_sum = cell_offsets_out[id3_out] + cell_offsets_in[id3_in]
        else:
            id3_atom_in = idx_in_s[id3_in]
            id3_atom_out = idx_out_s[id3_out]
            cell_offsets_sum = cell_offsets_out[id3_out] - cell_offsets_in[id3_in]
        mask = (id3_atom_in != id3_atom_out) | torch.any(cell_offsets_sum != 0, dim=-1)

        if return_adj:
            adj_edges = masked_select_sparsetensor_flat(adj_edges, mask)
            id3_in = adj_edges.storage.value().clone()
            id3_out = adj_edges.storage.row()
            res = [id3_in, id3_out, adj_edges]
        else:
            id3_in = id3_in[mask]
            id3_out = id3_out[mask]
            res = [id3_in, id3_out]

        if return_ragged:
            # id3_out has to be sorted for this
            id3_ragged_idx = get_ragged_idx(id3_out, dim_size=num_edges)
            res.append(id3_ragged_idx)

        return res

    def get_quadruplets(
        self,
        edge_index,
        edge_index_qint,
        cell_offsets,
        cell_offsets_qint,
        num_atoms,
    ):
        """
        Get all d->b for each edge c->a and connection b->a
        Careful about periodic images!
        Separate interaction cutoff not supported.

        Returns
        -------
        id3_db: torch.Tensor, shape (nTriplets,)
            Indices of input edge in triplet d->b->a.
        id3_qint_ba_abd: torch.Tensor, shape (nTriplets,)
            Interaction indices of output edge in triplet d->b->a.
        id3_ca: torch.Tensor, shape (nTriplets,)
            Indices of output edge in triplet c->a<-b.
        id3_qint_ba_cab: torch.Tensor, shape (nTriplets,)
            Interaction indices of input edge in triplet c->a<-b.
        id4_db: torch.Tensor, shape (nQuadruplets,)
            Indices of input edge d->b in quadruplet
        id4_ca: torch.Tensor, shape (nQuadruplets,)
            Indices of output edge c->a in quadruplet
        id3_to_id4_abd: torch.Tensor, shape (nQuadruplets,)
            Indices to map from triplet d->b->a to quadruplet d->b->a<-c.
        id3_to_id4_cab: torch.Tensor, shape (nQuadruplets,)
            Indices to map from triplet c->a<-b to quadruplet d->b->a<-c.
        id4_ragged_idx: torch.Tensor, shape (,)
            Indices enumerating the copies of id4_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index
        idx_qint_s, idx_qint_t = edge_index_qint
        # c->a (source=c, target=a)
        num_edges = idx_s.size(0)

        id3_db, id3_qint_ba_abd, adj_abd = self.get_mixed_triplets(
            edge_index,
            edge_index_qint,
            cell_offsets,
            cell_offsets_qint,
            num_atoms,
            to_outedge=True,
            return_adj=True,
        )
        # Triplets d->b->a

        id3_qint_ba_cab, id3_ca = self.get_mixed_triplets(
            edge_index_qint,
            edge_index,
            cell_offsets_qint,
            cell_offsets,
            num_atoms,
            to_outedge=False,
        )
        # Triplets c->a<-b

        # ---------------- Quadruplets -----------------
        # Repeat output indices by counting the number of input triplets
        # segment_coo assumes sorted id3_qint_ba_abd
        ones = id3_qint_ba_abd.new_ones(1).expand_as(id3_qint_ba_abd)
        num_trip_ba_out = segment_coo(
            ones, id3_qint_ba_abd, dim_size=idx_qint_s.size(0)
        )

        num_quad_cab = num_trip_ba_out[id3_qint_ba_cab]
        id4_ca = torch.repeat_interleave(id3_ca, num_quad_cab)
        id4_qint_ba = torch.repeat_interleave(id3_qint_ba_cab, num_quad_cab)
        id3_to_id4_cab = torch.repeat_interleave(
            torch.arange(len(id3_ca), device=idx_s.device, dtype=idx_s.dtype),
            num_quad_cab,
        )

        # Generate input indices by using the adjacency matrix adj_abd
        adj_abd.set_value_(
            torch.arange(len(id3_db), device=idx_s.device, dtype=idx_s.dtype),
            layout="coo",
        )
        adj_ca_abd = adj_abd[id3_qint_ba_cab]  # Rows are edges ba
        id3_to_id4_abd = adj_ca_abd.storage.value()
        id4_db = id3_db[id3_to_id4_abd]

        # Remove quadruplets with c == d
        # Triplets should already ensure that a != d and b != c
        # Compare atom indices and cell offsets
        idx_c = idx_s[id4_ca]
        idx_d = idx_s[id4_db]

        cell_offset_cd = (
            cell_offsets[id4_db] + cell_offsets_qint[id4_qint_ba] - cell_offsets[id4_ca]
        )
        mask_cd = (idx_c != idx_d) | torch.any(cell_offset_cd != 0, dim=-1)

        id4_ca = id4_ca[mask_cd]
        id3_to_id4_cab = id3_to_id4_cab[mask_cd]
        id3_to_id4_abd = id3_to_id4_abd[mask_cd]

        # id4_ca has to be sorted for this
        id4_ragged_idx = get_ragged_idx(id4_ca, dim_size=num_edges)

        return (
            id3_db,
            id3_qint_ba_abd,
            id3_ca,
            id3_qint_ba_cab,
            id4_ca,
            id3_to_id4_abd,
            id3_to_id4_cab,
            id4_ragged_idx,
        )

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def symmetrize_tensor(self, tensor, index, inverse_neg):
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor, sign * tensor])
        # Order and filter for unique edges
        tensor_ordered = tensor_cat[index]
        return tensor_ordered

    def symmetrize_edges(
        self,
        edge_index,
        cell_offsets,
        neighbors,
        batch_idx,
        reorder_tensors,
        reorder_tensors_invneg,
    ):
        """
        Symmetrize edges to ensure existence of counter-directional edges.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors.
        If `symmetric_edge_symmetrization` is False,
        we only use i->j edges here. So we lose some j->i edges
        and add others by making it symmetric.
        If `symmetric_edge_symmetrization` is True,
        we always use both directions.
        """
        num_atoms = batch_idx.shape[0]

        if self.symmetric_edge_symmetrization:
            edge_index_bothdir = torch.cat(
                [edge_index, edge_index.flip(0)],
                dim=1,
            )
            cell_offsets_bothdir = torch.cat(
                [cell_offsets, -cell_offsets],
                dim=0,
            )

            # Filter for unique edges
            edge_ids = get_edge_id(edge_index_bothdir, cell_offsets_bothdir, num_atoms)
            unique_ids, unique_inv = torch.unique(edge_ids, return_inverse=True)
            perm = torch.arange(
                unique_inv.size(0),
                dtype=unique_inv.dtype,
                device=unique_inv.device,
            )
            unique_idx = scatter(
                perm,
                unique_inv,
                dim=0,
                dim_size=unique_ids.shape[0],
                reduce="min",
            )
            edge_index_new = edge_index_bothdir[:, unique_idx]

            # Order by target index
            edge_index_order = torch.argsort(edge_index_new[1])
            edge_index_new = edge_index_new[:, edge_index_order]
            unique_idx = unique_idx[edge_index_order]

            # Subindex remaining tensors
            cell_offsets_new = cell_offsets_bothdir[unique_idx]
            reorder_tensors = [
                self.symmetrize_tensor(tensor, unique_idx, False)
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.symmetrize_tensor(tensor, unique_idx, True)
                for tensor in reorder_tensors_invneg
            ]

            # Count edges per image
            # segment_coo assumes sorted edge_index_new[1] and batch_idx
            ones = edge_index_new.new_ones(1).expand_as(edge_index_new[1])
            neighbors_per_atom = segment_coo(
                ones, edge_index_new[1], dim_size=num_atoms
            )
            neighbors_per_image = segment_coo(
                neighbors_per_atom, batch_idx, dim_size=neighbors.shape[0]
            )
        else:
            # Generate mask
            mask_sep_atoms = edge_index[0] < edge_index[1]
            # Distinguish edges between the same (periodic) atom by ordering the cells
            cell_earlier = (
                (cell_offsets[:, 0] < 0)
                | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
                | (
                    (cell_offsets[:, 0] == 0)
                    & (cell_offsets[:, 1] == 0)
                    & (cell_offsets[:, 2] < 0)
                )
            )
            mask_same_atoms = edge_index[0] == edge_index[1]
            mask_same_atoms &= cell_earlier
            mask = mask_sep_atoms | mask_same_atoms

            # Mask out counter-edges
            edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

            # Concatenate counter-edges after normal edges
            edge_index_cat = torch.cat(
                [edge_index_new, edge_index_new.flip(0)],
                dim=1,
            )

            # Count remaining edges per image
            batch_edge = torch.repeat_interleave(
                torch.arange(neighbors.size(0), device=edge_index.device),
                neighbors,
            )
            batch_edge = batch_edge[mask]
            # segment_coo assumes sorted batch_edge
            # Factor 2 since this is only one half of the edges
            ones = batch_edge.new_ones(1).expand_as(batch_edge)
            neighbors_per_image = 2 * segment_coo(
                ones, batch_edge, dim_size=neighbors.size(0)
            )

            # Create indexing array
            edge_reorder_idx = repeat_blocks(
                neighbors_per_image // 2,
                repeats=2,
                continuous_indexing=True,
                repeat_inc=edge_index_new.size(1),
            )

            # Reorder everything so the edges of every image are consecutive
            edge_index_new = edge_index_cat[:, edge_reorder_idx]
            cell_offsets_new = self.select_symmetric_edges(
                cell_offsets, mask, edge_reorder_idx, True
            )
            reorder_tensors = [
                self.select_symmetric_edges(tensor, mask, edge_reorder_idx, False)
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.select_symmetric_edges(tensor, mask, edge_reorder_idx, True)
                for tensor in reorder_tensors_invneg
            ]

        # Indices for swapping c->a and a->c (for symmetric MP)
        # To obtain these efficiently and without any index assumptions,
        # we get order the counter-edge IDs and then
        # map this order back to the edge IDs.
        # Double argsort gives the desired mapping
        # from the ordered tensor to the original tensor.
        edge_ids = get_edge_id(edge_index_new, cell_offsets_new, num_atoms)
        order_edge_ids = torch.argsort(edge_ids)
        inv_order_edge_ids = torch.argsort(order_edge_ids)
        edge_ids_counter = get_edge_id(
            edge_index_new.flip(0), -cell_offsets_new, num_atoms
        )
        order_edge_ids_counter = torch.argsort(edge_ids_counter)
        id_swap = order_edge_ids_counter[inv_order_edge_ids]

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_per_image,
            reorder_tensors,
            reorder_tensors_invneg,
            id_swap,
        )

    def subselect_edges(
        self,
        data,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        cutoff=None,
        max_neighbors=None,
    ):
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        if max_neighbors is not None:
            edge_mask, neighbors = get_max_neighbors_mask(
                natoms=data.natoms,
                index=edge_index[1],
                atom_distance=edge_dist,
                max_num_neighbors_threshold=max_neighbors,
            )
            if not torch.all(edge_mask):
                edge_index = edge_index[:, edge_mask]
                cell_offsets = cell_offsets[edge_mask]
                edge_dist = edge_dist[edge_mask]
                edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_graph(self, data, cutoff, max_neighbors):
        otf_graph = cutoff > 6 or max_neighbors > 50

        if self.use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, cutoff, max_neighbors
                )
            else:
                edge_index = data.edge_index
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=False,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            # These vectors actually point in the opposite direction.
            # But we want to use col as idx_t for efficient aggregation.
            edge_vector = -out["distance_vec"] / edge_dist[:, None]
            cell_offsets = -cell_offsets  # a - c + offset
        else:
            otf_graph = True
            edge_index = radius_graph(
                data.pos,
                r=self.cutoff,
                batch=data.batch,
                max_num_neighbors=max_neighbors,
            )
            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            edge_vector = -distance_vec / edge_dist[:, None]
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            neighbors = compute_neighbors(data, edge_index)

        # Mask interaction edges if required
        if otf_graph or np.isclose(cutoff, 6):
            select_cutoff = None
        else:
            select_cutoff = cutoff
        if otf_graph or max_neighbors == 50:
            select_neighbors = None
        else:
            select_neighbors = max_neighbors
        (
            edge_index,
            cell_offsets,
            neighbors,
            edge_dist,
            edge_vector,
        ) = self.subselect_edges(
            data=data,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            edge_dist=edge_dist,
            edge_vector=edge_vector,
            cutoff=select_cutoff,
            max_neighbors=select_neighbors,
        )

        return (
            edge_index,
            cell_offsets,
            neighbors,
            edge_dist,
            edge_vector,
        )

    def subselect_graph(
        self,
        data,
        cutoff,
        max_neighbors,
        cutoff_other,
        max_neighbors_other,
        edge_index_other,
        cell_offsets_other,
        neighbors_other,
        D_st_other,
        V_st_other,
    ):
        # Check if embedding edges are different from interaction edges
        if np.isclose(cutoff, cutoff_other):
            select_cutoff = None
        else:
            select_cutoff = cutoff
        if max_neighbors == max_neighbors_other:
            select_neighbors = None
        else:
            select_neighbors = max_neighbors

        (edge_index, cell_offsets, neighbors, D_st, V_st,) = self.subselect_edges(
            data=data,
            edge_index=edge_index_other,
            cell_offsets=cell_offsets_other,
            neighbors=neighbors_other,
            edge_dist=D_st_other,
            edge_vector=V_st_other,
            cutoff=select_cutoff,
            max_neighbors=select_neighbors,
        )

        return (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        )

    def get_graphs_and_indices(self, data):
        num_atoms = data.atomic_numbers.size(0)

        # Atom interaction graph is always the largest
        if (
            self.atom_edge_interaction
            or self.edge_atom_interaction
            or self.atom_interaction
        ):
            (
                edge_index_aint,
                cell_offsets_aint,
                neighbors_aint,
                D_aint_st,
                V_aint_st,
            ) = self.generate_graph(data, self.cutoff_aint, self.max_neighbors_aint)
            (edge_index, cell_offsets, neighbors, D_st, V_st,) = self.subselect_graph(
                data,
                self.cutoff,
                self.max_neighbors,
                self.cutoff_aint,
                self.max_neighbors_aint,
                edge_index_aint,
                cell_offsets_aint,
                neighbors_aint,
                D_aint_st,
                V_aint_st,
            )
            (
                edge_index_aeaint,
                cell_offsets_aeaint,
                _,
                D_aeaint_st,
                V_aeaint_st,
            ) = self.subselect_graph(
                data,
                self.cutoff_aeaint,
                self.max_neighbors_aeaint,
                self.cutoff_aint,
                self.max_neighbors_aint,
                edge_index_aint,
                cell_offsets_aint,
                neighbors_aint,
                D_aint_st,
                V_aint_st,
            )
        else:
            (
                edge_index,
                cell_offsets,
                neighbors,
                D_st,
                V_st,
            ) = self.generate_graph(data, self.cutoff, self.max_neighbors)
            edge_index_aint = (None, None)
            cell_offsets_aint = None
            neighbors_aint = None
            D_aint_st = None
            V_aint_st = None
            edge_index_aeaint = (None, None)
            cell_offsets_aeaint = None
            D_aeaint_st = None
            V_aeaint_st = None
        if self.quad_interaction:
            if (
                self.atom_edge_interaction
                or self.edge_atom_interaction
                or self.atom_interaction
            ):
                (
                    edge_index_qint,
                    cell_offsets_qint,
                    _,
                    D_qint_st,
                    V_qint_st,
                ) = self.subselect_graph(
                    data,
                    self.cutoff_qint,
                    self.max_neighbors_qint,
                    self.cutoff_aint,
                    self.max_neighbors_aint,
                    edge_index_aint,
                    cell_offsets_aint,
                    neighbors_aint,
                    D_aint_st,
                    V_aint_st,
                )
            else:
                assert self.cutoff_qint <= self.cutoff
                assert self.max_neighbors_qint <= self.max_neighbors
                (
                    edge_index_qint,
                    cell_offsets_qint,
                    _,
                    D_qint_st,
                    V_qint_st,
                ) = self.subselect_graph(
                    data,
                    self.cutoff_qint,
                    self.max_neighbors_qint,
                    self.cutoff,
                    self.max_neighbors,
                    edge_index,
                    cell_offsets,
                    neighbors,
                    D_st,
                    V_st,
                )

            # Only use quadruplets for certain tags
            self.qint_tags = self.qint_tags.to(edge_index_qint.device)
            tags_s = data.tags[edge_index_qint[0]]
            tags_t = data.tags[edge_index_qint[1]]
            qint_tag_mask_s = (tags_s[..., None] == self.qint_tags).any(dim=-1)
            qint_tag_mask_t = (tags_t[..., None] == self.qint_tags).any(dim=-1)
            qint_tag_mask = qint_tag_mask_s | qint_tag_mask_t
            edge_index_qint = edge_index_qint[:, qint_tag_mask]
            cell_offsets_qint = cell_offsets_qint[qint_tag_mask, :]
            D_qint_st = D_qint_st[qint_tag_mask]
            V_qint_st = V_qint_st[qint_tag_mask, :]
        else:
            edge_index_qint = (None, None)
            cell_offsets_qint = None
            D_qint_st = None
            V_qint_st = None

        # Symmetrize edges for swapping in symmetric message passing
        (
            edge_index,
            cell_offsets,
            neighbors,
            [D_st],
            [V_st],
            id_swap,
        ) = self.symmetrize_edges(
            edge_index, cell_offsets, neighbors, data.batch, [D_st], [V_st]
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms
        )

        # Additional indices for quadruplets
        if self.quad_interaction:
            (
                id3_db,
                id3_qint_ba_abd,
                id3_ca_q,
                id3_qint_ba_cab,
                id4_ca,
                id3_to_id4_abd,
                id3_to_id4_cab,
                id4_ragged_idx,
            ) = self.get_quadruplets(
                edge_index,
                edge_index_qint,
                cell_offsets,
                cell_offsets_qint,
                num_atoms,
            )
        else:
            id3_db, id3_qint_ba_abd = None, None
            id3_ca_q, id3_qint_ba_cab = None, None
            id4_ca = None
            id3_to_id4_abd, id3_to_id4_cab = None, None
            id4_ragged_idx = None

        if self.atom_edge_interaction:
            (
                id3_aeaint_ba_aeint,
                id3_ca_aeint,
                id3_ragged_idx_aeint,
            ) = self.get_mixed_triplets(
                edge_index_aeaint,
                edge_index,
                cell_offsets_aeaint,
                cell_offsets,
                num_atoms=num_atoms,
                return_ragged=True,
            )
        else:
            id3_aeaint_ba_aeint = None
            id3_ca_aeint = None
            id3_ragged_idx_aeint = None
        if self.edge_atom_interaction:
            (
                id3_ba_eaint,
                id3_aeaint_ca_eaint,
                id3_ragged_idx_eaint,
            ) = self.get_mixed_triplets(
                edge_index,
                edge_index_aeaint,
                cell_offsets,
                cell_offsets_aeaint,
                num_atoms=num_atoms,
                return_ragged=True,
            )
            # edge_index_aeaint has to be sorted for this
            idx_aeaint_ragged = get_ragged_idx(edge_index_aeaint[1], dim_size=num_atoms)
        else:
            id3_ba_eaint = None
            id3_aeaint_ca_eaint = None
            id3_ragged_idx_eaint = None
            idx_aeaint_ragged = None
        if self.atom_interaction:
            # edge_index_aint has to be sorted for this
            idx_aint_ragged = get_ragged_idx(edge_index_aint[1], dim_size=num_atoms)
        else:
            idx_aint_ragged = None

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            D_qint_st,
            V_qint_st,
            edge_index_aeaint,
            idx_aeaint_ragged,
            D_aeaint_st,
            V_aeaint_st,
            edge_index_aint,
            idx_aint_ragged,
            D_aint_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            id3_db,
            id3_qint_ba_abd,
            id3_ca_q,
            id3_qint_ba_cab,
            id4_ca,
            id3_to_id4_abd,
            id3_to_id4_cab,
            id4_ragged_idx,
            id3_aeaint_ba_aeint,
            id3_ca_aeint,
            id3_ragged_idx_aeint,
            id3_ba_eaint,
            id3_aeaint_ca_eaint,
            id3_ragged_idx_eaint,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            D_qint_st,
            V_qint_st,
            edge_index_aeaint,
            idx_aeaint_ragged,
            D_aeaint_st,
            V_aeaint_st,
            edge_index_aint,
            idx_aint_ragged,
            D_aint_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            id3_db,
            id3_qint_ba_abd,
            id3_ca_q,
            id3_qint_ba_cab,
            id4_ca,
            id3_to_id4_abd,
            id3_to_id4_cab,
            id4_ragged_idx,
            id3_aeaint_ba_aeint,
            id3_ca_aeint,
            id3_ragged_idx_aeint,
            id3_ba_eaint,
            id3_aeaint_ca_eaint,
            id3_ragged_idx_eaint,
        ) = self.get_graphs_and_indices(data)
        idx_s, idx_t = edge_index
        idx_aeaint_s, idx_aeaint_t = edge_index_aeaint
        idx_aint_s, idx_aint_t = edge_index_aint

        rbf = self.radial_basis(D_st)

        # Calculate triplet angles
        cosφ_cab = inner_product_clamped(V_st[id3_ca], V_st[id3_ba])
        rad_cbf_tint, sph_cbf_tint = self.cbf_basis_tint(D_st, cosφ_cab)

        if self.quad_interaction:
            # Calculate quadruplet angles
            cosφ_cab_q, cosφ_abd, angle_cabd = self.calculate_quad_angles(
                V_st,
                V_qint_st,
                id3_db,
                id3_qint_ba_abd,
                id3_ca_q,
                id3_qint_ba_cab,
                id3_to_id4_abd,
                id3_to_id4_cab,
            )

            rad_cbf_qint, sph_cbf_qint = self.cbf_basis_qint(D_qint_st, cosφ_abd)
            rad_sbf_qint, sph_sbf_qint = self.sbf_basis_qint(
                D_st,
                cosφ_cab_q[id3_to_id4_cab],
                angle_cabd,
            )
        if self.atom_edge_interaction:
            rbf_aeaint = self.radial_basis_aeaint(D_aeaint_st)
            cosφ_cab_aeint = inner_product_clamped(
                V_st[id3_ca_aeint], V_aeaint_st[id3_aeaint_ba_aeint]
            )
            rad_cbf_aeint, sph_cbf_aeint = self.cbf_basis_aeint(D_st, cosφ_cab_aeint)
        if self.edge_atom_interaction:
            cosφ_cab_eaint = inner_product_clamped(
                V_aeaint_st[id3_aeaint_ca_eaint], V_st[id3_ba_eaint]
            )
            rad_cbf_eaint, sph_cbf_eaint = self.cbf_basis_eaint(
                D_aeaint_st, cosφ_cab_eaint
            )
        if self.atom_interaction:
            rad_aint = self.radial_basis_aint(D_aint_st)

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        # Shared Down Projections
        if self.quad_interaction:
            rbf_qint = self.mlp_rbf_qint(rbf)
            cbf_qint = self.mlp_cbf_qint(
                rbf=rad_cbf_qint,
                sph=sph_cbf_qint,
                idx_sph=id3_qint_ba_abd,
            )
            sbf_qint = self.mlp_sbf_qint(
                rbf=rad_sbf_qint,
                sph=sph_sbf_qint,
                idx_sph=id4_ca,
                idx_ragged_sph=id4_ragged_idx,
            )
        else:
            rbf_qint = None
            cbf_qint = None
            sbf_qint = None

        if self.atom_edge_interaction:
            rbf_aeint = self.mlp_rbf_aeint(rbf_aeaint)
            cbf_aeint = self.mlp_cbf_aeint(
                rbf=rad_cbf_aeint,
                sph=sph_cbf_aeint,
                idx_sph=id3_ca_aeint,
                idx_ragged_sph=id3_ragged_idx_aeint,
            )
        else:
            rbf_aeint = None
            cbf_aeint = None
        if self.edge_atom_interaction:
            rbf_eaint = self.mlp_rbf_eaint(rbf)
            cbf_eaint = self.mlp_cbf_eaint(
                rbf=rad_cbf_eaint,
                sph=sph_cbf_eaint,
                idx_rad=idx_aeaint_t,
                idx_ragged_rad=idx_aeaint_ragged,
                idx_sph=id3_aeaint_ca_eaint,
                idx_ragged_sph=id3_ragged_idx_eaint,
                num_atoms=num_atoms,
            )
        else:
            rbf_eaint = None
            cbf_eaint = None
        if self.atom_interaction:
            rbf_aint = self.mlp_rbf_aint(
                rbf=rad_aint,
                idx_rad=idx_aint_t,
                idx_ragged_rad=idx_aint_ragged,
                num_atoms=num_atoms,
            )
        else:
            rbf_aint = None

        rbf_tint = self.mlp_rbf_tint(rbf)
        cbf_tint = self.mlp_cbf_tint(
            rbf=rad_cbf_tint,
            sph=sph_cbf_tint,
            idx_sph=id3_ca,
            idx_ragged_sph=id3_ragged_idx,
        )

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        # (nAtoms, num_targets), (nEdges, num_targets)

        if self.out_mlp_E is not None:
            E_t, F_st = [E_t], [F_st]

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf_qint=rbf_qint,
                cbf_qint=cbf_qint,
                sbf_qint=sbf_qint,
                rbf_tint=rbf_tint,
                cbf_tint=cbf_tint,
                rbf_aeint=rbf_aeint,
                cbf_aeint=cbf_aeint,
                rbf_eaint=rbf_eaint,
                cbf_eaint=cbf_eaint,
                rbf_aint=rbf_aint,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                id3_ragged_idx=id3_ragged_idx,
                id3_db=id3_db,
                id4_ca=id4_ca,
                id3_to_id4_abd=id3_to_id4_abd,
                id4_ragged_idx=id4_ragged_idx,
                id3_aeaint_ba_aeint=id3_aeaint_ba_aeint,
                id3_ca_aeint=id3_ca_aeint,
                id3_ragged_idx_aeint=id3_ragged_idx_aeint,
                id3_ba_eaint=id3_ba_eaint,
                id3_aeaint_ca_eaint=id3_aeaint_ca_eaint,
                id3_ragged_idx_eaint=id3_ragged_idx_eaint,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
                idx_aeaint_s=idx_aeaint_s,
                idx_aeaint_t=idx_aeaint_t,
                idx_aeaint_ragged=idx_aeaint_ragged,
                idx_aint_s=idx_aint_s,
                idx_aint_t=idx_aint_t,
                idx_aint_ragged=idx_aint_ragged,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            if self.out_mlp_E is not None:
                F_st.append(F)
                E_t.append(E)
            else:
                F_st += F
                E_t += E

        if self.out_mlp_E is not None:
            E_t = self.out_mlp_E(torch.cat(E_t, dim=-1))
            if self.direct_forces:
                F_st = self.out_mlp_F(torch.cat(F_st, dim=-1))
            with torch.cuda.amp.autocast(False):
                E_t = self.out_energy(E_t.float())
                if self.direct_forces:
                    F_st = self.out_forces(F_st.float())

        nMolecules = torch.max(batch) + 1
        if self.extensive:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
            )  # (nMolecules, num_targets)

        if self.regress_forces:
            if self.direct_forces:
                if self.forces_coupled:  # enforce F_st = F_ts
                    nEdges = idx_t.shape[0]
                    id_undir = repeat_blocks(
                        neighbors // 2, repeats=2, continuous_indexing=True
                    )
                    F_st = scatter(
                        F_st,
                        id_undir,
                        dim=0,
                        dim_size=int(nEdges / 2),
                        reduce="mean",
                    )  # (nEdges/2, num_targets)
                    F_st = F_st[id_undir]  # (nEdges, num_targets)

                # map forces in edge directions
                F_st_vec = F_st[:, :, None] * V_st[:, None, :]
                # (nEdges, num_targets, 3)
                F_t = scatter(
                    F_st_vec,
                    idx_t,
                    dim=0,
                    dim_size=num_atoms,
                    reduce="add",
                )  # (nAtoms, num_targets, 3)
            else:
                F_t = self.force_scaler.calc_forces_and_update(E_t, pos)

            E_t = E_t.squeeze(1)  # (num_molecules)
            F_t = F_t.squeeze(1)  # (num_atoms, 3)
            return E_t, F_t
        else:
            E_t = E_t.squeeze(1)  # (num_molecules)
            return E_t

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
