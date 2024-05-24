import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
links_df = pd.read_csv('links.csv')
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')
tags_df = pd.read_csv('tags.csv')

# Merge dataframes
data = pd.merge(ratings_df, movies_df, on='movieId')
tags_grouped = tags_df.groupby('movieId')['tag'].apply(list).reset_index()
data = pd.merge(data, tags_grouped, on='movieId', how='left')

# Replace NaN values in the 'tag' column with an empty list
data['tag'] = data['tag'].fillna('').apply(lambda x: ','.join(x) if isinstance(x, list) else '')

# Preprocess data
data['userId'] = data['userId'].astype(str)
data['movieId'] = data['movieId'].astype(str)

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define feature columns for wide part of the model
wide_feature_columns = [
    feature_column.categorical_column_with_hash_bucket('userId', hash_bucket_size=1000),
    feature_column.categorical_column_with_hash_bucket('movieId', hash_bucket_size=1000)
]

# One-hot encode the tags and create feature columns for deep part of the model
mlb = MultiLabelBinarizer()
mlb.fit(tags_df['tag'].apply(lambda x: [x]))

tag_columns = feature_column.categorical_column_with_vocabulary_list(
    'tag', mlb.classes_.tolist())

# Convert genres into categorical vocabulary
genres_vocab = movies_df['genres'].str.split('|').explode().unique().tolist()
genres_column = feature_column.categorical_column_with_vocabulary_list(
    'genres', genres_vocab)

deep_feature_columns = [
    feature_column.indicator_column(tag_columns),
    feature_column.indicator_column(genres_column)
]

# Define wide columns for memorization
wide_columns_for_memorization = [
    feature_column.crossed_column(['userId', 'movieId'], hash_bucket_size=int(1e4))
]

# Define deep columns for generalization
deep_columns_for_generalization = deep_feature_columns

# Create input function
def input_fn(df, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), df['rating']))
    dataset = dataset.batch(batch_size)
    return dataset

# Define model
model = tf.estimator.DNNLinearCombinedRegressor(
    linear_feature_columns=wide_feature_columns + wide_columns_for_memorization,
    dnn_feature_columns=deep_columns_for_generalization,
    dnn_hidden_units=[128, 64],
    dnn_activation_fn=tf.nn.relu,
    dnn_dropout=0.2
)

# Train model
model.train(input_fn=lambda: input_fn(train_data), steps=1000)

# Evaluate model
predictions = model.predict(input_fn=lambda: input_fn(test_data))
y_true = test_data['rating'].values
y_pred = np.array([prediction['predictions'][0] for prediction in predictions])

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
print("RMSE:", rmse)

# Compute NDCG@K
def compute_ndcg(predictions, true_ratings, k):
    # Sort predictions in descending order and take top k
    sorted_indices = np.argsort(predictions)[::-1][:k]
    sorted_true_ratings = true_ratings[sorted_indices]
    
    # Compute Discounted Cumulative Gain (DCG)
    dcg = sorted_true_ratings[0]
    for i in range(1, k):
        dcg += sorted_true_ratings[i] / np.log2(i + 1)
    
    # Compute Ideal Discounted Cumulative Gain (IDCG)
    ideal_sorted_true_ratings = np.sort(true_ratings)[::-1][:k]
    idcg = ideal_sorted_true_ratings[0]
    for i in range(1, k):
        idcg += ideal_sorted_true_ratings[i] / np.log2(i + 1)
    
    # Compute NDCG
    if idcg == 0:  # Handle division by zero
        return 0.0
    else:
        return dcg / idcg

# Compute NDCG@5
ndcg_5 = compute_ndcg(y_pred, y_true, 5)
print("NDCG@5:", ndcg_5)