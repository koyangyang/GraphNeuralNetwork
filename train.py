import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader

transform = T.ToUndirected()  # Add reverse edge types.
data = OGB_MAG(root='./dataset', preprocess='metapath2vec', transform=transform)[0]

train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors={key: [15] * 2 for key in data.edge_types},
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=('paper', data['paper'].train_mask),
)

batch = next(iter(train_loader))
print(batch)
