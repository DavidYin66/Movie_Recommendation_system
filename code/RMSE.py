import pandas as pd
import numpy as np

# 载入 Movielens 数据集的 ratings 数据
ratings = pd.read_csv('D:/机器学习/期末大作业/ml-latest-small/ratings.csv')

# 计算所有评分的均值
mean_rating = ratings['rating'].mean()

# 计算差值的平方
squared_differences = (ratings['rating'] - mean_rating) ** 2

# 计算RMSE
rmse = np.sqrt(squared_differences.mean())

print("根均方差 (RMSE):", rmse)
