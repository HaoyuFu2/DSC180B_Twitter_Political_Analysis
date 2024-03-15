import warnings
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
 

def load_and_merge_data(labels_file, tweets_file): #-- Checked
    """
    Load tweet data and labels from specified CSV files and merge them on user_id.
    
    Args:
        labels_file (str): Path to the CSV file containing user IDs and labels.
        tweets_file (str): Path to the CSV file containing user IDs and cleaned tweets.
    
    Returns:
        pd.DataFrame: Merged DataFrame with each user's tweets aggregated.
    """

    labels_df = pd.read_csv(labels_file)
    tweets_df = pd.read_csv(tweets_file)[['user_id', 'cleaned_tweets']]
    merged_df = pd.merge(tweets_df, labels_df, on='user_id', how='inner'
                         )
    merged_df['all_tweets'] = merged_df.groupby('user_id'
            )['cleaned_tweets'].transform(lambda x: ' '.join(x))
    return merged_df.drop_duplicates(subset='user_id'
            ).reset_index(drop=True)


def split_dataset(df, test_size=0.8, random_state=42): #-- Checked
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): The dataset to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
        
    Returns:
        tuple: The split data (X_train, X_test, Y_train, Y_test).
    """

    X = df['all_tweets']
    Y = df[[
        'label',
        'subjectivity',
#         'sentiment',
#         'binary_strong_negation',
#         'binary_excessive_marks',
#         'binary_topic_diversity',
        ]]
    return train_test_split(X, Y, test_size=test_size,
                            random_state=random_state)


def vectorize_text( #-- Checked
    X_train,
    X_test,
    max_features=5000,
    min_df=5,
    max_df=0.8,
    ):
    """
    Convert a collection of text documents to a matrix of TF-IDF features.
    
    Args:
        X_train (array-like): Training text data.
        X_test (array-like): Testing text data.
        max_features (int): Maximum number of features to extract.
        min_df (int): When building the vocabulary, ignore terms that have a document frequency strictly lower than the given threshold.
        max_df (float): When building the vocabulary, ignore terms that have a document frequency strictly higher than the given threshold.
        
    Returns:
        tuple: Vectorized training and testing data, and the vectorizer instance.
    """

    vectorizer = TfidfVectorizer(max_features=max_features,
                                 min_df=min_df, max_df=max_df)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return (X_train_tfidf, X_test_tfidf, vectorizer)


def reduce_dimensions(X_train_tfidf, X_test_tfidf, n_components=300):
    """
    Perform dimensionality reduction on the feature set using Truncated SVD.
    
    Args:
        X_train_tfidf (sparse matrix): TF-IDF matrix for the training data.
        X_test_tfidf (sparse matrix): TF-IDF matrix for the testing data.
        n_components (int): Number of dimensions to reduce to.
        
    Returns:
        tuple: Dimensionality-reduced training and testing data, and the SVD instance.
    """

    svd = TruncatedSVD(n_components=n_components)
    X_train_reduced = svd.fit_transform(X_train_tfidf)
    X_test_reduced = svd.transform(X_test_tfidf)
    return (X_train_reduced, X_test_reduced, svd)


def train_and_predict(X_train_reduced, Y_train, X_test_reduced):
    """
    Train a multi-output logistic regression model and predict the testing set.
    
    Args:
        X_train (array-like): Training feature set.
        Y_train (array-like): Training target set.
        X_test (array-like): Testing feature set.
        
    Returns:
        tuple: Trained model and predictions for the testing set.
    """

    model = MultiOutputClassifier(LogisticRegression(max_iter=10000))
    model.fit(X_train_reduced, Y_train)
    return (model, model.predict(X_test_reduced))


def evaluate_predictions(Y_test, Y_pred):
    """
    Print the classification report and accuracy for each label in the testing set.
    
    Args:
        Y_test (pd.DataFrame): True labels for the testing set.
        Y_pred (array-like): Predicted labels for the testing set.
    """
    print(f"Label: {Y_test.columns[0]}")
    print(classification_report(Y_test.iloc[:, 0], Y_pred[:, 0]))
    print('Accuracy:', accuracy_score(Y_test.iloc[:, 0], Y_pred[:, 0]))

    

def predict_new_data(
    tweets_file,
    vectorizer,
    svd,
    model,
    label_columns = ['label', 'subjectivity'],
    ):
    """
    Predict new data using the provided model, vectorizer, and SVD transformer.

    Args:
        tweets_file (str): Path to the CSV file containing new cleaned tweets.
        vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
        svd (TruncatedSVD): Trained SVD instance.
        model (MultiOutputClassifier): Trained multi-output classifier.
        label_columns (list): List of label names corresponding to model's outputs.

    Returns:
        pd.DataFrame: New tweets with predicted labels and features.
    """

    new_tweets_df = pd.read_csv(tweets_file)
    new_tweets_df['all_tweets'] = new_tweets_df['cleaned_tweets'
            ].fillna('')
    X_new_tfidf = vectorizer.transform(new_tweets_df['all_tweets'])
    X_new_reduced = svd.transform(X_new_tfidf)
    predicted_labels = model.predict(X_new_reduced)

    for (i, label) in enumerate(label_columns):
        new_tweets_df[label] = predicted_labels[:, i]

    return new_tweets_df


def convert_labels(df, label_columns): #-- Checked
    """
    Convert string labels in the DataFrame to numeric labels, 
    with special handling for labels 'b' and 't' which are converted 
    to 1 and -1 respectively.
    
    Args:
        df (pd.DataFrame): The DataFrame with labels to convert.
        label_columns (list): List of columns in df that contain labels to be converted.
        
    Returns:
        tuple: Updated DataFrame with numeric labels.
    """

    df[label_columns] = df[label_columns].replace({'b': 1, 't': -1})
        
    return df
