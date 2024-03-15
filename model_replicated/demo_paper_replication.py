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

# load data
df = pd.read_parquet('data/gss.parquet')

# encode labels
le = LabelEncoder()
df['yearid_id'] = le.fit_transform(df['yearid'])
df['question_id'] = le.fit_transform(df['variable'])
df['year'] = df['yearid'] // 10000
df['year_order'] = le.fit_transform(df['year'])

# split dataset
(train_data, val_data) = train_test_split(df, test_size=0.1)
train_data.to_parquet('data/train_data.parquet')
val_data.to_parquet('data/val_data.parquet')

# make embeddings
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
question_list = df[['question_id', 'question'
                   ]].drop_duplicates().sort_values('question_id'
        )['question'].tolist()
embeddings = []
for question in tqdm(question_list):
    prompt = \
        '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

'''
    prompt += f"### Instruction:{question}\n\n### Response:"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    outputs = model(input_ids, output_hidden_states=True)
    embeddings.append(outputs.hidden_states[-1][:, -1, :
                      ].detach().numpy())
pickle.dump(embeddings, open('gpt2_embeddings.pkl', 'wb'))

# build the models
def get_model(dim=50):
    weights = np.vstack(pickle.load(open('gpt2_embeddings.pkl', 'rb')))
    individual_id = Input(name='individual_id', shape=(1, ))
    question_id = Input(name='question_id', shape=(1, ))
    year_id = Input(name='year_id', shape=(1, ))

    # first embedding: individual belief (the binary responses)
    x1 = Embedding(df['yearid_id'].max() + 1, dim, name='individual_embedding')(individual_id)
    x1 = Reshape((dim, ), name='reshape1')(x1)

    # second embedding: questions
    question_embedding = Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=False, name='question_embedding')(question_id)
    x2 = Dense(dim, name='question_embedding2')(question_embedding) 
    x2 = Reshape((dim, ), name='reshape2')(x2)

    # third embeddings: the time (year) of the response
    x3 = Embedding(df['year_order'].max() + 1, dim, name='year_embedding')(year_id)
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
    inputs = {'individual_id': individual_id, 'question_id': question_id, 'year_id': year_id}
    model = Model(inputs=inputs, outputs=out)
    return model

# model training
def train(train, val):
    model = get_model()
    # make the features for training and validating sets
    train_features = {
        'individual_id': np.array(train['yearid_id']),
        'question_id': np.array(train['question_id']),
        'year_id': np.array(train['year_order'])
    }
    val_features = {
        'individual_id': np.array(val['yearid_id']),
        'question_id': np.array(val['question_id']),
        'year_id': np.array(val['year_order'])
    }
    train_label = np.array(train['binarized'])
    val_label = np.array(val['binarized'])

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

# run the models
train_data = pd.read_parquet('data/train_data.parquet')
val_data = pd.read_parquet('data/val_data.parquet')
trained_model, training_history = train(train_data, val_data)
trained_model.save_weights('data/missing_imputation.h5')

# model evaluations
prediction_model = get_model()
prediction_model.load_weights('data/missing_imputation.h5')
val_features = {
    'individual_id': np.array(val_data['yearid_id']),
    'question_id': np.array(val_data['question_id']),
    'year_id': np.array(val_data['year_order'])
}
pred = prediction_model.predict(val_features)
obs = val_data['binarized'].values
roc_auc = roc_auc_score(obs, pred)
accuracy = accuracy_score(obs, pred > 0.5)
print(f"ROC AUC Score: {roc_auc:.3f}")
print(f"Accuracy Score: {accuracy:.3f}")
