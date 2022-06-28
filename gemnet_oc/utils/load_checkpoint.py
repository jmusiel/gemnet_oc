import torch


def load_checkpoint(ocp_class, checkpoint_path, cpu=True):
    if cpu:
        map_location = torch.device("cpu")
    else:
        map_location = torch.device(f"cuda:{0}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    config = checkpoint["config"]["model_attributes"]
    model = ocp_class(num_atoms=0, bond_feat_dim=0, num_targets=1, **config)

    first_key = next(iter(checkpoint["state_dict"]))
    if first_key.split(".")[0] == "module":
        new_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
        if next(iter(new_dict)).split(".")[0] == "module":
            new_dict = {k[7:]: v for k, v in new_dict.items()}
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(checkpoint["state_dict"])
    return model