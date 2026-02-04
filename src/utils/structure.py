from torch_geometric.data import Data

class ProteinData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'scalars':
            return 0  # This is the "correct" way for graph-level features
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'scalars':
            return 0 # Do not increment values for graph-level features
        return super().__inc__(key, value, *args, **kwargs)