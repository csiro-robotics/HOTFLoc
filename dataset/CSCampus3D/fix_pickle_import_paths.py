"""
Script to repair the import paths for TrainingTuple objects in campus3D training
pickles.
"""
import sys
import pickle
import dataset
from dataset.base_datasets import TrainingTuple
sys.modules['datasets'] = dataset
# sys.modules['datasets.base_datasets.TrainingTuple'] = TrainingTuple

if __name__ == "__main__":
    root_path = "/scratch3/gri317/datasets/cs_campus3d/benchmark_datasets/"
    train_file = root_path + "training_queries_umd_4096_MinkLoc3Dv2_format.pickle"
    with open(train_file, 'rb') as f:
        train_queries = pickle.load(f)
    
    # Remove ref to old path
    del sys.modules['datasets']

    # Save queries with correct path to class
    train_queries_fixed = {}
    for key, query in train_queries.items():
        train_tuple = TrainingTuple(
            id=query.id, timestamp=query.timestamp,
            rel_scan_filepath=query.rel_scan_filepath, positives=query.positives,
            non_negatives=query.non_negatives, position=query.position 
        )
        train_queries_fixed[key] = train_tuple

    # Save pickles
    save_file = root_path + "training_queries_umd_4096_MinkLoc3Dv2_virga_format.pickle"
    with open(save_file, 'wb') as handle:
        pickle.dump(train_queries_fixed, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pass