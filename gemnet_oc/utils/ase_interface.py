import torch
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets.lmdb_dataset import data_list_collater
import torch_geometric


class ASEInterface:
    def __init__(
        self,
        cutoff: float = 6,
        max_neighbors: int = 50,
    ):

        self.a2g_predict = AtomsToGraphs(
            max_neigh=max_neighbors,
            radius=cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
        )

        self.a2g_train = AtomsToGraphs(
            max_neigh=max_neighbors,
            radius=cutoff,
            r_energy=True,
            r_forces=True,
            r_distances=True,
            r_edges=False,
        )

    def a2g_convert(self, atoms, train: bool):
        if train:
            data_object = self.a2g_train.convert(atoms)
        else:
            data_object = self.a2g_predict.convert(atoms)

        if not hasattr(data_object, "tags"):
            data_object.tags = torch.ones(data_object.num_nodes)

        return data_object

    def get_data_from_atoms(self, atoms_list):
        graphs_list = [self.a2g_convert(atoms, False) for atoms in atoms_list]
        batch_list = [data_list_collater([i_graph], otf_graph=True) for i_graph in graphs_list]
        return batch_list

    def get_collated_data_from_atoms(self, atoms_list):
        """
        Deprecated function, turns atoms list into one batch, 
        so forward function returns one big tensor for each forward pass.
        More efficient in some cases, since combines the forward passes
        But unhelpful when we want to separate each image into its own forward pass
        """
        graphs_list = [self.a2g_convert(atoms, False) for atoms in atoms_list]
        data_loader = data_list_collater(graphs_list, otf_graph=True)

        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        data_list = []
        for batch_list in data_loader:
            for batch in batch_list:
                data_list.append(batch)

        return data_list

