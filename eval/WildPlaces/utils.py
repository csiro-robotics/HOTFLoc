import os 
import pickle 
import torch 
import numpy as np
from misc.utils import TrainingParams
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader
from misc.optional_deps import lazy

# Lazy-load MinkowskiEngine - will return real module or helpful stub
ME = lazy("MinkowskiEngine", feature="sparse convolutions")

def query_to_timestamp(query):
    base = os.path.basename(query)
    timestamp = float(base.replace('.pcd', ''))
    return timestamp

def euclidean_dist(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()

def cosine_dist(query, database):
    return np.array(1 - torch.einsum('D,ND->N', torch.tensor(query), torch.tensor(database)))

def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        file = pickle.load(f)
    return file 

def get_latent_vectors(model, data_set, device, params: TrainingParams):
    # Adapted from original PointNetVLAD code
    pc_loader = AboveUnderPointCloudLoader()

    model.eval()
    embeddings = None
    for i, elem_ndx in enumerate(data_set):
        pc_file_path = os.path.join(params.dataset_folder, data_set[elem_ndx]["query"])
        pc = pc_loader(pc_file_path)
        pc = torch.tensor(pc)

        embedding = compute_embedding(model, pc, device, params)
        if embeddings is None:
            embeddings = np.zeros((len(data_set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings

def compute_embedding(model, pc, device, params: TrainingParams):
    coords, _ = params.model_params.quantizer(pc)
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        # Compute global descriptor
        y = model(batch, global_only=True)
        embedding = y['global'].detach().cpu().numpy()

    return embedding