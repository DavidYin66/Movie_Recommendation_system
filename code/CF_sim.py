import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# 读取电影信息和评分数据
movies = pd.read_csv('D:/机器学习/期末大作业/ml-latest-small/movies.csv')
ratings = pd.read_csv('D:/机器学习/期末大作业/ml-latest-small/ratings.csv')

# 查看数据
print(movies.head())
print(ratings.head())

# 构建用户-电影评分矩阵
user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# 计算电影之间的相似度矩阵（使用余弦相似度）
movie_similarity = cosine_similarity(user_movie_ratings.T)

# 将相似度矩阵转换为DataFrame，便于查看
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

# 为某个用户推荐电影
def recommend_movies_for_user(user_id, num_recommendations=5):
    # 获取用户评分过的电影
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # 获取用户评分过的电影的ID
    rated_movie_ids = user_ratings['movieId'].tolist()
    
    # 计算每个电影的推荐评分
    movie_scores = {}
    
    for movie_id in rated_movie_ids:
        # 获取该电影与所有其他电影的相似度
        similar_movies = movie_similarity_df[movie_id]
        
        # 获取用户对该电影的评分
        movie_rating = user_ratings[user_ratings['movieId'] == movie_id]['rating'].values[0]
        
        # 更新每个未评分电影的评分
        for other_movie_id in movie_similarity_df.columns:
            if other_movie_id not in rated_movie_ids:
                if other_movie_id not in movie_scores:
                    movie_scores[other_movie_id] = 0
                movie_scores[other_movie_id] += similar_movies[other_movie_id] * movie_rating
    
    # 按照预测评分排序，获取前num_recommendations个电影
    recommended_movie_ids = sorted(movie_scores, key=movie_scores.get, reverse=True)[:num_recommendations]
    
    # 获取电影标题
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    
    return recommended_movies

# 为测试集中的每一条数据计算RMSE
def calculate_rmse():
    # 读取测试集
    testset = ratings.sample(frac=0.0002, random_state=43) 
    
    true_ratings = []
    predicted_ratings = []
    
    for index, row in testset.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        true_rating = row['rating']
        
        # 预测评分
        similar_movies = movie_similarity_df[movie_id]
        
        # 获取用户评分过的电影
        user_ratings = ratings[ratings['userId'] == user_id]
        rated_movie_ids = user_ratings['movieId'].tolist()
        
        # 计算预测评分
        predicted_rating = 0
        total_similarity = 0
        
        for rated_movie_id in rated_movie_ids:
            if rated_movie_id != movie_id:
                similarity = similar_movies[rated_movie_id]
                rating = user_ratings[user_ratings['movieId'] == rated_movie_id]['rating'].values[0]
                predicted_rating += similarity * rating
                total_similarity += abs(similarity)
        
        if total_similarity > 0:
            predicted_rating /= total_similarity
        
        true_ratings.append(true_rating)
        predicted_ratings.append(predicted_rating)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return rmse


# 为用户 1 推荐电影
recommended_movies = recommend_movies_for_user(user_id=1, num_recommendations=5)
print(recommended_movies[['movieId', 'title', 'genres']])

# 计算并输出RMSE
rmse = calculate_rmse()
print(f'RMSE: {rmse}')
