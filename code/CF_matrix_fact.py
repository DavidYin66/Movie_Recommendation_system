import pandas as pd

# 读取电影信息和评分数据
movies = pd.read_csv('D:\机器学习\期末大作业\ml-latest-small\movies.csv')
ratings = pd.read_csv('D:\机器学习\期末大作业\ml-latest-small\\ratings.csv')

# 查看数据
print(movies.head())
print(ratings.head())

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# 准备数据
reader = Reader(rating_scale=(1, 5))  # 评分范围是1到5
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 拆分数据为训练集和测试集
trainset, testset = train_test_split(data, test_size=0.2)

# 使用SVD（矩阵分解）算法
algo = SVD()

# 训练模型
algo.fit(trainset)

# 在测试集上进行预测
predictions = algo.test(testset)

# 计算RMSE（均方根误差）
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# 为某个用户推荐电影
def recommend_movies_for_user(user_id, num_recommendations=5):
    # 获取用户评分过的电影
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # 获取用户评分过的电影的ID
    rated_movie_ids = user_ratings['movieId'].tolist()
    
    # 获取所有电影的ID
    all_movie_ids = movies['movieId'].tolist()
    
    # 找出用户未评分的电影ID
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    # 对未评分电影进行预测
    predictions = [algo.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    
    # 按照预测评分排序，获取前num_recommendations个电影
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    recommended_movie_ids = [prediction.iid for prediction in predictions[:num_recommendations]]
    
    # 获取电影标题
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    
    return recommended_movies

# 为用户 1 推荐电影
recommended_movies = recommend_movies_for_user(user_id=1, num_recommendations=5)
print(recommended_movies[['movieId', 'title', 'genres']])
