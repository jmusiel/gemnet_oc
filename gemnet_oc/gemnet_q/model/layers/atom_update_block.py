import math

import torch
from torch_scatter import scatter

from ..initializers import get_initializer
from .base_layers import Dense, ResidualLayer
from .scaling import ScaledModule, ScalingFactor


class AtomUpdateBlock(ScaledModule):
    """
    Aggregate the message embeddings of the atoms

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Name of the activation function to use in the dense layers.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation=None,
    ):
        super().__init__()

        self.dense_rbf = Dense(
            emb_size_rbf, emb_size_edge, activation=None, bias=False
        )
        self.scale_sum = ScalingFactor()

        self.layers = self.get_mlp(
            emb_size_edge, emb_size_atom, nHidden, activation
        )

    def get_mlp(self, units_in, units, nHidden, activation):
        if units_in != units:
            dense1 = Dense(units_in, units, activation=activation, bias=False)
            mlp = [dense1]
        else:
            mlp = []
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(nHidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]

        mlp_rbf = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * mlp_rbf

        x2 = scatter(
            x, id_j, dim=0, dim_size=nAtoms, reduce="sum"
        )  # (nAtoms, emb_size_edge)
        x = self.scale_sum(x2, x_ref=m)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)

        return x


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        num_targets: int
            Number of targets.
        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
        direct_forces: bool
            If true directly predict forces without taking the gradient of the energy potential.
        output_init: str
            Initialization method for the final dense layer.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        num_targets: int,
        activation=None,
        direct_forces=True,
        output_init: str = "HeOrthogonal",
        output_emb: bool = False,
    ):

        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
        )

        self.output_init = output_init
        self.direct_forces = direct_forces
        self.output_emb = output_emb

        self.seq_energy_pre = self.layers  # inherited from parent class
        if not output_emb:
            self.out_energy = Dense(
                emb_size_atom, num_targets, bias=False, activation=None
            )
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(
                emb_size_atom, emb_size_atom, nHidden_afteratom, activation
            )
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        else:
            self.seq_energy2 = None

        if self.direct_forces:
            self.scale_rbf_F = ScalingFactor()
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, nHidden, activation
            )
            if not output_emb:
                self.out_forces = Dense(
                    emb_size_edge, num_targets, bias=False, activation=None
                )
            self.dense_rbf_F = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )

        self.reset_parameters()

    def reset_parameters(self):
        if not self.output_emb:
            initializer = get_initializer(self.output_init)
            self.out_energy.reset_parameters(initializer)
            if self.direct_forces:
                self.out_forces.reset_parameters(initializer)

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            (E, F): tuple
            - E: torch.Tensor, shape=(nAtoms, num_targets)
            - F: torch.Tensor, shape=(nEdges, num_targets)
            Energy and force prediction
        """
        nAtoms = h.shape[0]

        # -------------------------------------- Energy Prediction -------------------------------------- #
        rbf_emb_E = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * rbf_emb_E

        x_E = scatter(
            x, id_j, dim=0, dim_size=nAtoms, reduce="sum"
        )  # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(x_E, x_ref=m)

        for layer in self.seq_energy_pre:
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        if self.seq_energy2 is not None:
            x_E = x_E + h
            x_E = x_E * self.inv_sqrt_2
            for layer in self.seq_energy2:
                x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        if not self.output_emb:
            with torch.cuda.amp.autocast(False):
                x_E = self.out_energy(x_E.float())  # (nAtoms, num_targets)

        # --------------------------------------- Force Prediction -------------------------------------- #
        if self.direct_forces:
            x_F = m
            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            rbf_emb_F = self.dense_rbf_F(rbf)  # (nEdges, emb_size_edge)
            x_F_rbf = x_F * rbf_emb_F
            x_F = self.scale_rbf_F(x_F_rbf, x_ref=x_F)

            if not self.output_emb:
                with torch.cuda.amp.autocast(False):
                    x_F = self.out_forces(x_F.float())  # (nEdges, num_targets)
        else:
            x_F = 0
        # ----------------------------------------------------------------------------------------------- #

        return x_E, x_F
