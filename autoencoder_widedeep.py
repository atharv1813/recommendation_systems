import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

links_df = pd.read_csv('links.csv')
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')
tags_df = pd.read_csv('tags.csv')

data = pd.merge(ratings_df, movies_df, on='movieId')
tags_grouped = tags_df.groupby('movieId')['tag'].apply(list).reset_index()
data = pd.merge(data, tags_grouped, on='movieId', how='left')

data['tag'] = data['tag'].fillna('').apply(lambda x: ','.join(x) if isinstance(x, list) else '')

data['userId'] = data['userId'].astype(str)
data['movieId'] = data['movieId'].astype(str)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

wide_feature_columns = [
    feature_column.categorical_column_with_hash_bucket('userId', hash_bucket_size=1000),
    feature_column.categorical_column_with_hash_bucket('movieId', hash_bucket_size=1000)
]

mlb = MultiLabelBinarizer()
mlb.fit(tags_df['tag'].apply(lambda x: [x]))

tag_columns = feature_column.categorical_column_with_vocabulary_list(
    'tag', mlb.classes_.tolist())

genres_vocab = movies_df['genres'].str.split('|').explode().unique().tolist()
genres_column = feature_column.categorical_column_with_vocabulary_list(
    'genres', genres_vocab)

deep_feature_columns = [
    feature_column.indicator_column(genres_column)
]

wide_columns_for_memorization = [
    feature_column.crossed_column(['userId', 'movieId'], hash_bucket_size=int(1e4))
]

deep_columns_for_generalization = deep_feature_columns

def input_fn(df, encoded_tags, batch_size=32):
    features = dict(df)
    features['encoded_tags'] = encoded_tags
    labels = df['rating']
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)
    return dataset

model_without_encoders = tf.estimator.DNNLinearCombinedRegressor(
    linear_feature_columns=wide_feature_columns + wide_columns_for_memorization,
    dnn_feature_columns=deep_columns_for_generalization,
    dnn_hidden_units=[128, 64],
    dnn_activation_fn=tf.nn.relu,
    dnn_dropout=0.2
)

model_without_encoders.train(input_fn=lambda: input_fn(train_data, None), steps=1000)

predictions_without_encoders = model_without_encoders.predict(input_fn=lambda: input_fn(test_data, None))
y_pred_without_encoders = np.array([prediction['predictions'][0] for prediction in predictions_without_encoders])

rmse_without_encoders = np.sqrt(np.mean((y_pred_without_encoders - test_data['rating'].values) ** 2))

input_dim = len(mlb.classes_)
encoding_dim = 64

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.models.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(mlb.transform(tags_df['tag'].apply(lambda x: [x])), 
                mlb.transform(tags_df['tag'].apply(lambda x: [x])),
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_split=0.2)

encoder = tf.keras.models.Model(input_layer, encoded)

encoded_tags_train = encoder.predict(mlb.transform(train_data['tag'].apply(lambda x: [x])))
encoded_tags_test = encoder.predict(mlb.transform(test_data['tag'].apply(lambda x: [x])))

deep_feature_columns.append(feature_column.numeric_column('encoded_tags', shape=[encoding_dim]))

model_with_encoders = tf.estimator.DNNLinearCombinedRegressor(
    linear_feature_columns=wide_feature_columns + wide_columns_for_memorization,
    dnn_feature_columns=deep_columns_for_generalization,
    dnn_hidden_units=[128, 64],
    dnn_activation_fn=tf.nn.relu,
    dnn_dropout=0.2
)

model_with_encoders.train(input_fn=lambda: input_fn(train_data, encoded_tags_train), steps=1000)

predictions_with_encoders = model_with_encoders.predict(input_fn=lambda: input_fn(test_data, encoded_tags_test))
y_pred_with_encoders = np.array([prediction['predictions'][0] for prediction in predictions_with_encoders])

rmse_with_encoders = np.sqrt(np.mean((y_pred_with_encoders - test_data['rating'].values) ** 2))

print("RMSE without autoencoders:", rmse_without_encoders)
print("RMSE with autoencoders:", rmse_with_encoders)


def compute_ndcg(predictions, true_ratings, k):
    sorted_indices = np.argsort(predictions)[::-1][:k]
    sorted_true_ratings = true_ratings[sorted_indices]
    
    dcg = sorted_true_ratings[0]
    for i in range(1, k):
        dcg += sorted_true_ratings[i] / np.log2(i + 1)
    
    ideal_sorted_true_ratings = np.sort(true_ratings)[::-1][:k]
    idcg = ideal_sorted_true_ratings[0]
    for i in range(1, k):
        idcg += ideal_sorted_true_ratings[i] / np.log2(i + 1)
    
    if idcg == 0:  
        return 0.0
    else:
        return dcg / idcg

ndcg_5 = compute_ndcg(y_pred_with_encoders , test_data['rating'].values, 5)
print("NDCG@5 for autoencoders:", ndcg_5)

ndcg_5 = compute_ndcg(y_pred_without_encoders , test_data['rating'].values, 5)
print("NDCG@5 without autoencoders:", ndcg_5)