import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

# 数据预处理
data = {
    'user_id': [1, 2, 1, 3, 4, 2, 5, 4, 5, 3],
    'item_id': [101, 101, 102, 103, 104, 102, 105, 104, 101, 103],
    'rating': [5, 3, 2, 5, 4, 5, 3, 5, 4, 1]
}
df = pd.DataFrame(data)

# 划分数据集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 定义Dataset
class RatingDataset(Dataset):
    def __init__(self, dataframe):
        self.user_ids = dataframe['user_id'].values
        self.item_ids = dataframe['item_id'].values
        self.ratings = dataframe['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx],self.ratings[idx]

# 创建DataLoader
train_dataset = RatingDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 模型定义
class EmbeddingModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        return torch.sum(user_embedding * item_embedding, dim=1)

# 参数设定
num_users = df['user_id'].max() + 1  # 加1确保索引不越界
num_items = df['item_id'].max() + 1  # 同上
embedding_dim = 10 # 经验决定 embedding 维度

# 实例化模型和优化器
model = EmbeddingModel(num_users, num_items, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10):
    model.train()
    for user_ids, item_ids, ratings in train_loader:
        print(user_ids)
        print(ratings)
        # 将numpy数组转换为Tensor
        user_ids = torch.LongTensor(user_ids)
        item_ids = torch.LongTensor(item_ids)
        ratings = torch.LongTensor(ratings).float()

        # 前向传播和损失计算
        predictions = model(user_ids, item_ids)
        loss = nn.MSELoss()(predictions, ratings)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 2. 使用 Embedding 召回
# 计算特定用户对所有物品的偏好
user_id = 1
user_embedding = model.user_embedding(torch.LongTensor([user_id]))
item_embeddings = model.item_embedding.weight
scores = torch.matmul(item_embeddings, user_embedding.t()).squeeze() # 点积得到相关性得分

# 根据分数高低召回物品
recommended_items_score, recommended_items = torch.topk(scores, k=5)
print("召回：")
print(recommended_items_score.float(), recommended_items)

# 3. 使用 Embedding 粗排和精排
# 构建一个带有embedding输入的神经网络模型
class RankingModel(nn.Module):
    def __init__(self, user_embedding, item_embedding):
        super(RankingModel, self).__init__()
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.fc_layers = nn.Sequential(
            nn.Linear(20, 10),  # embedding_dim * 2 -> intermediate dimensions
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        output = self.fc_layers(x)
        return output

# 使用这个模型进行精排
ranking_model = RankingModel(model.user_embedding, model.item_embedding)

# 训练模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ranking_model.parameters(), lr=0.01)

# 训练循环
for epoch in range(5):
    for user_ids, item_ids, ratings  in train_loader:
        optimizer.zero_grad()
        outputs = ranking_model(user_ids, item_ids)
        loss = criterion(outputs.squeeze(), ratings)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")


 # 假设我们有一些待评分的用户和物品的ID
user_ids_to_predict = torch.tensor([0, 1])  # 这里只是示例ID
item_ids_to_predict = torch.tensor([1, 2])

# 设置模型为评估模式
ranking_model.eval()

# 禁用梯度计算，因为在预测时不需要梯度
with torch.no_grad():
    # 进行预测
    predictions = ranking_model(user_ids_to_predict, item_ids_to_predict)
    print("Predicted Scores:", predictions)