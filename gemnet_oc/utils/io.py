from ase.io import iread, write
from os.path import isdir
from os import makedirs
import pickle

def attach_results(atoms_list, **kwargs):
    for i, image in enumerate(atoms_list):
        for key, value in kwargs.items():
            image.calc.results[key] = value[i]
    return atoms_list

def latent_inf_to_xyz(model, original_filepath: str, new_filepath: str=None, batch_size=None):
    # if not given, make new_filepath a renamed version of original file, in a subfolder
    if new_filepath is None:
        split_path = original_filepath.split("/")
        filename_split = split_path[-1].split(".")
        filename = filename_split[0]
        for split in filename_split[1:-1]:
            filename += "." + split
        filename += "_latent.xyz"
        new_filepath = split_path[0]
        for split in split_path[1:-1]:
            new_filepath += "/" + split
        new_filepath += "/latent_rep/" + filename

    # if path to new file doesn't exist, create it
    new_filepath_split = new_filepath.split("/")
    new_dir = new_filepath_split[0]
    for split in new_filepath_split[1:-1]:
        new_dir += "/" + split
    if not isdir(new_dir):
        makedirs(new_dir)

    # if batch size is none, read in, run inference, and save, one at a time
    if batch_size is None:
        append = False
        for image in iread(original_filepath):
            res_list, lat_list = model.get_latent_and_residuals([image], [image.get_potential_energy()])
            finished_images = attach_results([image], residual=res_list, latent_rep=lat_list)
            write(new_filepath, finished_images, append=append)
            append = True
    # otherwise read in a batch, run inference on each, then save the whole batch
    else:
        image_list = []
        finished_images = []
        append = False
        for i, image in enumerate(iread(original_filepath)):
            image_list.append(image)
            if i % batch_size == 0:
                res_list, lat_list = model.get_latent_and_residuals(image_list, [im.get_potential_energy() for im in image_list])
                finished_images += attach_results(image_list, residual=res_list, latent_rep=lat_list)
                write(new_filepath, finished_images, append=append)
                image_list = []
                finished_images = []
                append = True
        if len(finished_images) > 0:
            write(new_filepath, finished_images, append=append)

def latent_inf_to_pickle(model, original_filepath: str, new_filepath: str=None, batch_size=None):
    # if not given, make new_filepath a renamed version of original file, in a subfolder
    if new_filepath is None:
        split_path = original_filepath.split("/")
        filename_split = split_path[-1].split(".")[0]
        filename = filename_split[0]
        for split in filename_split[1:-1]:
            filename += "." + split
        filename += "_latent.json"
        new_filepath = split_path[0]
        for split in split_path[1:-1]:
            new_filepath += "/" + split
        new_filepath += "/latent_rep/" + filename

    # if path to new file doesn't exist, create it
    new_filepath_split = new_filepath.split("/")
    new_dir = new_filepath_split[0]
    for split in new_filepath_split[1:-1]:
        new_dir += "/" + split
    if not isdir(new_dir):
        makedirs(new_dir)

    # if batch size is none, read in, run inference, and save, one at a time
    if batch_size is None:
        append = False
        with open(new_filepath, "wb") as f:
            for image in iread(original_filepath):
                res_list, lat_list = model.get_latent_and_residuals([image], [image.get_potential_energy()])
                results_pair = (res_list[0], lat_list[0])
                pickle.dump(results_pair, f)

    # otherwise read in a batch, run inference on each, then save the whole batch
    else:
        image_list = []
        finished_images = []
        with open(new_filepath, "wb") as f:
            for i, image in enumerate(iread(original_filepath)):
                image_list.append(image)
                if i % batch_size == 0:
                    res_list, lat_list = model.get_latent_and_residuals(image_list, [im.get_potential_energy() for im in image_list])
                    for r, l in zip(res_list, lat_list):
                        pickle.dump((r, l), f)
                    image_list = []
            if len(image_list) > 0:
                res_list, lat_list = model.get_latent_and_residuals(image_list, [im.get_potential_energy() for im in image_list])
                for r, l in zip(res_list, lat_list):
                    pickle.dump((r, l), f)
