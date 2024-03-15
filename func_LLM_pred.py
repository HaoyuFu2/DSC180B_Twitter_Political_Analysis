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
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import Constant
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
    """
    Constructs and returns a model based on the provided dataframe's structure and dimensions.

    Parameters:
    - df: DataFrame containing the dataset. It is used to determine the maximum values for user_id and year_id embeddings.
    - dim: The dimensionality of the embeddings. Defaults to 50.

    Returns:
    - A compiled TensorFlow Keras model ready for training.
    """

    # Load pre-trained embedding weights

    weights = np.vstack(pickle.load(open('gpt2_embeddings.pkl', 'rb')))

    # Define input layers

    user_id = Input(name='user_id', shape=(1, ))
    year_id = Input(name='year_id', shape=(1, ))
    tweet_id = Input(name='tweet_id', shape=(1, ))
    subjectivity_input = Input(name='subjectivity', shape=(1, ),
                               dtype='float32')

    # User embedding

    user_embedding = Embedding(df['user_id'].max() + 1, dim,
                               name='user_embedding')(user_id)
    user_embedding = Reshape((dim, ), name='reshape_user'
                             )(user_embedding)

    # Tweet embedding (using pre-trained weights)

    tweet_embedding = Embedding(input_dim=weights.shape[0],
                                output_dim=weights.shape[1],
                                embeddings_initializer=Constant(weights),
                                trainable=False, name='tweet_embedding'
                                )(tweet_id)
    tweet_embedding = Dense(dim, name='dense_tweet')(tweet_embedding)
    tweet_embedding = Reshape((dim, ), name='reshape_tweet'
                              )(tweet_embedding)

    # Year embedding

    year_embedding = Embedding(df['year_id'].max() + 1, dim,
                               name='year_embedding')(year_id)
    year_embedding = Reshape((dim, ), name='reshape_year'
                             )(year_embedding)

    # Combine features

    combined_features = Concatenate(name='concat_features'
                                    )([user_embedding, tweet_embedding,
            year_embedding, subjectivity_input])

    # Network layers

    x = Dense(dim * 3, activation='relu', name='dense_1'
              )(combined_features)
    x = Dropout(0.2, name='dropout_1')(x)
    x = Dense(dim * 3, activation='relu', name='dense_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)

    # Output layer

    output = Dense(1, activation='sigmoid', name='output')(x)

    # Build and return model

    model = Model(inputs=[user_id, tweet_id, year_id,
                  subjectivity_input], outputs=output)
    return model

def train(df_train, df_val, model):
    """
    Trains the model using the provided training and validation datasets.

    Parameters:
    - df_train: DataFrame containing the training data.
    - df_val: DataFrame containing the validation data.
    - model: The TensorFlow Keras model to be trained.

    Returns:
    - model: The trained model.
    - history: Training history object containing details about the training process.
    """

    # Convert string labels to numeric labels

    le = LabelEncoder()
    df_train['label'] = le.fit_transform(df_train['label'])
    df_val['label'] = le.transform(df_val['label'])

    # Prepare training and validation data

    train_features = {
        'user_id': np.array(df_train['user_id'], dtype='int32'),
        'tweet_id': np.array(df_train['tweet_id'], dtype='int32'),
        'year_id': np.array(df_train['year_id'], dtype='int32'),
        'subjectivity': np.array(df_train['subjectivity'],
                                 dtype='float32'),
        }
    train_labels = np.array(df_train['label'], dtype='int32')

    val_features = {
        'user_id': np.array(df_val['user_id'], dtype='int32'),
        'tweet_id': np.array(df_val['tweet_id'], dtype='int32'),
        'year_id': np.array(df_val['year_id'], dtype='int32'),
        'subjectivity': np.array(df_val['subjectivity'], dtype='float32'
                                 ),
        }
    val_labels = np.array(df_val['label'], dtype='int32')

    # Compile the model

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model

    history = model.fit(train_features, train_labels,
                        validation_data=(val_features, val_labels),
                        epochs=10, batch_size=32)

    return (model, history)


def run_model(file_path='data/labels.parquet', size=1000):
    """
    Runs the entire model workflow including data loading, preprocessing, model training, and evaluation.

    Parameters:
    - file_path: String, the file path to the dataset stored in Parquet format.
    - size: Integer, the number of samples to load from the dataset. Default is 100.

    Returns:
    - training_history: A History object generated by the Keras model's fit method. Contains details about the training process.
    """

    # Load data

    df = \
        pd.read_parquet(file_path)[:size].drop_duplicates(subset=['tweet'
            ])

    # Encode labels

    le = LabelEncoder()
    df['user_id'] = le.fit_transform(df['user_id'])
    df['year_id'] = le.fit_transform(df['year'])
    df['tweet_id'] = le.fit_transform(df['tweet'])
    df['label'] = le.fit_transform(df['label'])  # Convert 'label' to numeric

    # Split dataset

    (train_data, val_data) = train_test_split(df, test_size=0.1)

    # Generate tweet embeddings

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model_gpt2 = AutoModelForCausalLM.from_pretrained('gpt2')
    tweet_list = df['tweet'].tolist()
    embeddings = []

    for tweet in tqdm(tweet_list, total=len(tweet_list)):
        prompt = f"### Tweet:{tweet}\n\n### Response:"
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        outputs = model_gpt2(input_ids, output_hidden_states=True)
        embeddings.append(outputs.hidden_states[-1][:, -1, :
                          ].detach().numpy())
    pickle.dump(embeddings, open('gpt2_embeddings.pkl', 'wb'))

    # Reinitialize model to ensure it's in the correct state for training

    trained_model = get_model(df)  # Assume get_model is defined elsewhere

    # Train the model

    (trained_model, training_history) = train(train_data, val_data,
            trained_model)

    # Save model weights with the correct file extension

    trained_model.save_weights('data/weights.weights.h5')

    # Load weights for evaluation

    prediction_model = get_model(df)  # Reinitialize model for evaluation
    prediction_model.load_weights('data/weights.weights.h5')

    # Prepare features for prediction

    val_features = {
        'user_id': np.array(val_data['user_id']),
        'tweet_id': np.array(val_data['tweet_id']),
        'year_id': np.array(val_data['year_id']),
        'subjectivity': np.array(val_data['subjectivity']),
        }

    # Predict and evaluate
    pred_probs = prediction_model.predict(val_features)
    pred_labels = (pred_probs > 0.5).astype(int)
    accuracy = accuracy_score(val_data['label'].values, pred_labels)
    print(f"Accuracy Score: {accuracy:.3f}")
    
    return prediction_model, training_history
