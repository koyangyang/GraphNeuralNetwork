import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero  # 核心模块
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import warnings

warnings.filterwarnings("ignore")

# Load the entire movie data from into memory
movies_df = pd.read_csv('dataset/ml-latest-small/movies.csv', index_col='movieId')

# Split genres and convert into indicaor variables:
genres = movies_df['genres'].str.get_dummies('|')

# Use genres as movie input features:
movie_feat = torch.from_numpy(genres.values).to(torch.float)
assert movie_feat.size() == (len(movies_df), 20)

# Load the entire rating data into memory
ratings_df = pd.read_csv('dataset/ml-latest-small/ratings.csv')
unique_user_id = ratings_df['userId'].unique()
unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedId': pd.RangeIndex(len(unique_user_id)),
})
unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedId': pd.RangeIndex(len(unique_movie_id)),
})
ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id, left_on='userId', right_on='userId', how='left')
ratings_user_id = torch.from_numpy(ratings_user_id['mappedId'].values)
ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id, left_on='movieId', right_on='movieId', how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedId'].values)

edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
assert edge_index_user_to_movie.size() == (2, len(ratings_df))

data = HeteroData()  # 创建异构图
# Save node indices:
data['user'].node_id = torch.arange(len(unique_user_id))
data['movie'].node_id = torch.arange(len(movies_df))
'''
当user的特征没有时（只有User_id）--> NLP任务
NLP:先把一句话做分词，将每个词转换成对应的ID，然后再把每个词的ID给它Embedding到固定的一个维度，得到一个特征
此任务：随机对user初始化一个特征（Embedding 矩阵），然后再训练过程中，通过GCN去学习user的特征
'''
# Add the node features and edge indices:
data['movie'].x = movie_feat
data['user', 'rates', 'movie'].edge_index = edge_index_user_to_movie
# 给边添加特征
data['user', 'rates', 'movie'].edge_attr = ratings_df[['rating', 'timestamp']]

data = T.ToUndirected()(data)  # 转换为无向图
transform = T.RandomLinkSplit(  # 生成标签
    num_val=0.1,  # 10%的数据作为验证集
    num_test=0.1,  # 10%的数据作为测试集
    disjoint_train_ratio=0.3,  # 监督边占训练集的比例，消极边占训练集的比例为1-0.3=0.7
    neg_sampling_ratio=2.0,  # 负采样比例，通过生成负样本与正样本进行有监督训练
    add_negative_train_samples=False,  # 是否将负样本加入训练集
    edge_types=('user', 'rates', 'movie'),
    rev_edge_types=('movie', 'rev_rates', 'user'),
)
train_data, valid_data, test_data = transform(data)

edge_label_index = train_data['user', 'rates', 'movie'].edge_label_index
edge_label = train_data['user', 'rates', 'movie'].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],  # 采样子图，在当前节点的一阶邻居选20个，二阶邻居选10个
    neg_sampling_ratio=2.0,  # 随机负样本的比例
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_index[0]]
        edge_feat_movie = x_movie[edge_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Model, self).__init__()
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            'user': self.user_emb(data["user"].node_id),
            'movie': self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(x_dict['user'], x_dict['movie'], data['user', 'rates', 'movie'].edge_label_index)
        return pred


model = Model(hidden_channels=64)
model.load_state_dict(torch.load('result/model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# epochs = 20
# for epoch in range(epochs):
#     total_loss = total_examples = 0
#     # for sampled_data in tqdm(train_loader):
#     for sampled_data in train_loader:
#         optimizer.zero_grad()
#         sampled_data = sampled_data.to(device)
#         pred = model(sampled_data)
#         ground_truth = sampled_data['user', 'rates', 'movie'].edge_label
#         loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * pred.numel()
#         total_examples += pred.numel()
#     print(f"Epoch {epoch} train loss: {total_loss / total_examples:.4f}")

model.eval()
edge_label_index = valid_data['user', 'rates', 'movie'].edge_label_index
edge_label = valid_data['user', 'rates', 'movie'].edge_label
val_loader = LinkNeighborLoader(
    data=valid_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=False,
)
total_loss = total_examples = 0
for sampled_data in val_loader:
    sampled_data = sampled_data.to(device)
    pred = model(sampled_data)
    ground_truth = sampled_data['user', 'rates', 'movie'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
val_loss = total_loss / total_examples
val_auc = (pred > 0).eq(ground_truth > 0).sum().item() / ground_truth.numel()
print(f"Validation loss: {val_loss:.4f}, AUC score: {val_auc:.4f}")

model.eval()
edge_label_index = test_data['user', 'rates', 'movie'].edge_label_index
edge_label = test_data['user', 'rates', 'movie'].edge_label
test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=False,
)
total_loss = total_examples = 0
for sampled_data in test_loader:
    sampled_data = sampled_data.to(device)
    pred = model(sampled_data)
    ground_truth = sampled_data['user', 'rates', 'movie'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
test_loss = total_loss / total_examples
test_auc = (pred > 0).eq(ground_truth > 0).sum().item() / ground_truth.numel()
print(f"Test loss: {test_loss:.4f}, AUC score: {test_auc:.4f}")
