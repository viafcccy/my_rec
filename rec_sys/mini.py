# 1. 模拟数据
import pandas as pd
from sklearn.model_selection import train_test_split

# 模拟数据：用户ID、物品ID、评分
data = {
    'user_id': [1, 2, 1, 3, 4, 2, 5, 4, 5, 3],
    'item_id': [101, 101, 102, 103, 104, 102, 105, 104, 101, 103],
    'rating': [5, 3, 2, 5, 4, 5, 3, 5, 4, 1]
}
df = pd.DataFrame(data)

# 划分数据集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(train_df)
print(test_df)

# 2. 召回
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 创建物品-用户矩阵
item_user_matrix = train_df.pivot_table(index='item_id', columns='user_id', values='rating', fill_value=0)
item_user_matrix = item_user_matrix.T

# 使用KNN进行物品相似度计算
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3, n_jobs=-1)
model_knn.fit(item_user_matrix)

# 召回示例：为用户1召回物品
distances, indices = model_knn.kneighbors(item_user_matrix.loc[1].values.reshape(1, -1), n_neighbors=3)
recalled_items = item_user_matrix.columns[indices.flatten()].tolist()

print(recalled_items)


# 3. 粗排
from sklearn.linear_model import Ridge

# 训练一个简单的线性模型进行评分预测
X_train = train_df[['user_id', 'item_id']]
y_train = train_df['rating']

# 模型训练
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)

# 预测召回物品的评分
X_test = pd.DataFrame({'user_id': [1]*len(recalled_items), 'item_id': recalled_items})
predicted_ratings = model_ridge.predict(X_test)

print(predicted_ratings)

# 4. 精排
from sklearn.ensemble import GradientBoostingRegressor

# 训练精排模型
model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_gbr.fit(X_train, y_train)

# 使用精排模型预测评分
final_ratings = model_gbr.predict(X_test)
print(final_ratings)

# 5. 反馈
# 假设有新的用户反馈
new_feedback = {'user_id': [1], 'item_id': [101], 'rating': [4]}
new_feedback_df = pd.DataFrame(new_feedback)

# 更新训练数据
train_df = pd.concat([train_df, new_feedback_df])

# 重新训练模型
model_gbr.fit(train_df[['user_id', 'item_id']], train_df['rating'])