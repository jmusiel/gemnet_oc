
import torch
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (conditional_grad)
from torch_scatter import scatter

from gemnet_oc.gemnet_q.model.utils import (inner_product_clamped, repeat_blocks)
from gemnet_oc.gemnet_q.model.gemnet import GemNet
import torch
from gemnet_oc.utils.ase_interface import ASEInterface


@registry.register_model("latent_gemnet_dev")
class LatentGemNet(GemNet):
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
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.interface = ASEInterface(
            kwargs.get("cutoff", 12.0),
            kwargs.get("max_neighbors", 100),
        )

    def get_latent_and_residuals(self, atoms_list, energy_ground_truths):
        h_list = []
        m_list = []
        E_list = []
        F_list = []
        data_list = self.interface.get_data_from_atoms(atoms_list)
        i = 0
        for data in data_list:
            i+=1
            E_t, F_t = self.forward(data)
            h, m = self.latent_h, self.latent_m
            self.latent_h = self.latent_m = None
            h_list.append(h)
            m_list.append(m)
            E_list.append(E_t)
            F_list.append(F_t)

        residual_list = []
        for E, y in zip(E_list, energy_ground_truths):
            residual_list.append(y-E)
        return h_list, residual_list

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.latent_h = None
        self.latent_m = None

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

        self.latent_h = h
        self.latent_m = m

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

