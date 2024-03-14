import os
import pickle
import torch
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from tensorflow.keras import Model
import tensorflow_recommenders as tfrs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Reshape

# set seeds in order to get consistent random results
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
tqdm.pandas()

# build the models
def get_model(df, dim=50):
    weights = np.vstack(pickle.load(open('gpt2_embeddings.pkl', 'rb')))
    user_id = Input(name='user_id', shape=(1, ))
    year_id = Input(name='year_id', shape=(1, ))
    tweet_id = Input(name='tweet_id', shape=(1, ))

    # first embedding: user's label
    x1 = Embedding(df['user_id'].max() + 1, dim, name='user_embedding')(user_id)
    x1 = Reshape((dim, ), name='reshape1')(x1)

    # second embedding: tweet
    tweet_embedding = Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=False, name='tweet_embedding')(tweet_id)
    x2 = Dense(dim, name='tweet_embedding2')(tweet_embedding) 
    x2 = Reshape((dim, ), name='reshape2')(x2)

    # third embeddings: year
    x3 = Embedding(df['year_id'].max() + 1, dim, name='year_embedding')(year_id)
    x3 = Reshape((dim, ), name='reshape3')(x3)

    # combine the above three embeddings
    x = [x1, x2, x3]
    x = tf.concat(x, axis=1, name='concat1')
    for i in range(3):
        x = tfrs.layers.dcn.Cross(projection_dim=dim*3, kernel_initializer="glorot_uniform", name=f'cross_layer_{i}')(x)
        x = Dropout(0.2)(x)
    for i in range(3):
        x = Dense(dim*3, activation="relu", name=f'dense_layer_{i}')(x)
        x = Dropout(0.2)(x)
    
    # generate predictions
    out = Dense(1, activation='sigmoid', name="out")(x)

    # model building
    inputs = {'user_id': user_id, 'tweet_id': tweet_id, 'year_id': year_id}
    model = Model(inputs=inputs, outputs=out)
    return model

# model training
def train(df, train, val):
    model = get_model(df)
    # make the features for training and validating sets
    train_features = {
        'user_id': np.array(train['user_id']),
        'tweet_id': np.array(train['tweet_id']),
        'year_id': np.array(train['year_id'])
    }
    val_features = {
        'user_id': np.array(val['user_id']),
        'tweet_id': np.array(val['tweet_id']),
        'year_id': np.array(val['year_id'])
    }
    train_label = np.array(train['label'])
    val_label = np.array(val['label'])

    # model compiling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=2e-5,
        decay_steps=80000,
        decay_rate=0.96,
        staircase=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(),
            tf.keras.metrics.BinaryAccuracy()
        ]
    )
    history = model.fit(
        train_features,
        train_label,
        validation_data=(val_features, val_label),
        batch_size=8,
        epochs=10,
        verbose=1,
        use_multiprocessing=True,
        workers=20
    )

    return model, history

def run_model(file_path='data/labels.parquet', size=1000):

    # load data
    df = pd.read_parquet(file_path)[:size].drop_duplicates(subset=['tweet'])

    # encode labels
    le = LabelEncoder()
    df['user_id'] = le.fit_transform(df['user_id'])
    df['year_id'] = le.fit_transform(df['year'])
    df['tweet_id'] = le.fit_transform(df['tweet'])

    # split dataset
    (train_data, val_data) = train_test_split(df, test_size=0.1)
    train_data.to_parquet('data/train_data.parquet')
    val_data.to_parquet('data/val_data.parquet')

    # make embeddings
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tweet_list = df['tweet'].drop_duplicates().tolist()
    embeddings = []

    for tweet in tqdm(tweet_list):
        prompt = \
            '''Below is a Tweet that discusses the Presidential Election. Write a response that appropriately describes the author's political position.
    '''
        prompt += f"### Tweet:{tweet}\n\n### Response:"
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        outputs = model(input_ids, output_hidden_states=True)
        embeddings.append(outputs.hidden_states[-1][:, -1, :].detach().numpy())
    pickle.dump(embeddings, open('gpt2_embeddings.pkl', 'wb'))

    # run the models
    train_data = pd.read_parquet('data/train_data.parquet')
    val_data = pd.read_parquet('data/val_data.parquet')
    trained_model, training_history = train(df, train_data, val_data)
    trained_model.save_weights('data/weights.h5')

    # model evaluations
    prediction_model = get_model(df)
    prediction_model.load_weights('data/weights.h5')
    val_features = {
        'user_id': np.array(val_data['user_id']),
        'tweet_id': np.array(val_data['tweet_id']),
        'year_id': np.array(val_data['year_id'])
    }
    pred = prediction_model.predict(val_features)
    obs = val_data['label'].values
    # roc_auc = roc_auc_score(obs, pred)
    accuracy = accuracy_score(obs, pred > 0.5)
    # print(f"ROC AUC Score: {roc_auc:.3f}")
    print(f"Accuracy Score: {accuracy:.3f}")

    return training_history

