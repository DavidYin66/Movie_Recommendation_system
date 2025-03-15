import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# 读取电影信息和评分数据
movies = pd.read_csv('D:/机器学习/期末大作业/ml-latest-small/movies.csv')
ratings = pd.read_csv('D:/机器学习/期末大作业/ml-latest-small/ratings.csv')

# 使用电影标题和类别创建 TF-IDF 特征
movies['genres'] = movies['genres'].str.split('|').apply(lambda x: ' '.join(x))  # 将多个类别合并成一个字符串

# 创建 TF-IDF 向量
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# 计算电影之间的余弦相似度
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 获取与某部电影相似的电影
def get_similar_movies(movie_id, cosine_sim=cosine_sim, top_n=5):
    # 获取电影的索引
    idx = movies[movies['movieId'] == movie_id].index[0]
    
    # 获取该电影与所有其他电影的相似度
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 按照相似度进行排序
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 获取相似度最高的前top_n部电影
    sim_scores = sim_scores[1:top_n+1]
    
    # 获取推荐的电影ID
    movie_indices = [i[0] for i in sim_scores]
    
    # 返回电影标题
    recommended_movies = movies.iloc[movie_indices]
    
    return recommended_movies[['movieId', 'title', 'genres']]

# 为电影ID 1 推荐相似电影
recommended_movies = get_similar_movies(movie_id=1, top_n=5)
print("Recommended Movies:")
print(recommended_movies)

# Step 1: 创建用户-电影评分矩阵
movie_avg_ratings = ratings.groupby('movieId')['rating'].mean()
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_movie_matrix = user_movie_matrix.apply(lambda x: x.fillna(0), axis=0)


print(user_movie_matrix.head())

# Step 2: 预测评分
def predict_ratings(user_movie_matrix, cosine_sim):
    # 将pred_matrix改为pandas的DataFrame对象
    pred_matrix = pd.DataFrame(np.zeros(user_movie_matrix.shape), columns=user_movie_matrix.columns, index=user_movie_matrix.index)

    for user_id in user_movie_matrix.index:
        for movie_id in user_movie_matrix.columns:
            if user_movie_matrix.at[user_id, movie_id] == 0:  # 只对未评分的电影进行预测
                # 获取与当前电影最相似的电影的评分
                # 需要确保 movie_id - 1 是有效的索引
                movie_idx = movie_id - 1  # cosine_sim 是 0-based index
                if movie_idx >= 0 and movie_idx < cosine_sim.shape[0]:  # 确保索引在有效范围内
                    sim_scores = cosine_sim[movie_idx]  # 获取与当前电影的相似度
                    weighted_ratings = 0
                    total_sim = 0
                    for other_movie_id in user_movie_matrix.columns:
                        if user_movie_matrix.at[user_id, other_movie_id] > 0:  # 用户已经评分的电影
                            other_movie_idx = other_movie_id - 1
                            if other_movie_idx >= 0 and other_movie_idx < cosine_sim.shape[0]:  # 确保索引有效
                                weighted_ratings += sim_scores[other_movie_idx] * user_movie_matrix.at[user_id, other_movie_id]
                                total_sim += abs(sim_scores[other_movie_idx])

                    if total_sim != 0:
                        pred_matrix.loc[user_id, movie_id] = weighted_ratings / total_sim

    return pred_matrix

# 获取预测评分矩阵
predicted_ratings = predict_ratings(user_movie_matrix, cosine_sim)

print(predicted_ratings.head())


# Step 3: 计算 RMSE
def compute_rmse(test_data_matrix, predicted_ratings):
    errors = []

    for user_id in test_data_matrix.index:
        for movie_id in test_data_matrix.columns:
            if test_data_matrix.at[user_id, movie_id] > 0:  # 只对已评分的电影计算误差
                actual_rating = test_data_matrix.at[user_id, movie_id]
                predicted_rating = predicted_ratings.at[user_id, movie_id]
                error = actual_rating - predicted_rating
                errors.append(error ** 2)

    rmse = np.sqrt(np.mean(errors))
    return rmse

# 从 ratings 中拆分出测试集和训练集（20% 为测试集）
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# 创建测试集的用户-电影评分矩阵
user_movie_test_matrix = test_data.pivot(index='userId', columns='movieId', values='rating')

# 计算每个电影的平均评分
movie_means = user_movie_test_matrix.mean(axis=0)

# 将电影的平均评分广播到矩阵中，以填充 NaN 值
user_movie_test_matrix = user_movie_test_matrix.apply(lambda col: col.fillna(movie_means[col.name]), axis=0)

# 计算 RMSE
rmse = compute_rmse(user_movie_test_matrix, predicted_ratings)
print(f"RMSE: {rmse}")






