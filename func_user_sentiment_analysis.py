import re
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def load_data(labels_file, tweets_file):
    """
    Load and preprocess tweet and label datasets.
    
    Args:
        labels_file (str): Path to the CSV file containing user IDs and labels and other binary sentiment labels.
        tweets_file (str): Path to the CSV file containing user IDs and cleaned tweets.
    
    Returns:
        pd.DataFrame: Merged DataFrame with aggregated tweets for each user and associated labels.
    
    Features:
        - 'user_id': Identifier for each user.
        - 'label': Political affiliation label ('biden' or 'trump').
        - 'all_tweets': Aggregated tweets for each user.
    """

    df = pd.read_csv(labels_file)[['user_id', 'label']]
    tweet = pd.read_csv(tweets_file)[['user_id', 'cleaned_tweets']]
    df['label'] = df['label'].replace({'b': 'biden', 't': 'trump'})
    merged_df = pd.merge(tweet, df, on='user_id', how='inner')
    merged_df['all_tweets'] = merged_df.groupby('user_id'
            )['cleaned_tweets'].transform(lambda x: ' '.join(x))
    return merged_df.drop_duplicates(subset='user_id'
            ).reset_index(drop=True)


def binary_sentiment_subjectivity(df, sentiment_threshold=0.1,
                                  subjectivity_threshold=0.5):
    """
    Calculate sentiment and subjectivity for each tweet and classify them based on the threshold.
    
    Args:
        df (pd.DataFrame): DataFrame with a column 'all_tweets' containing the text to analyze.
        threshold (float): Threshold value to classify sentiment and subjectivity.
        
    Updates:
        The input dataframe is updated to include 'sentiment' and 'subjectivity' columns.
    """

    def calculate_sentiment(tweets):
        sentiment = TextBlob(tweets).sentiment
        return {'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity}

    df['sentiment_details'] = df['all_tweets'
                                 ].apply(calculate_sentiment)
    df['subjectivity'] = df['sentiment_details'].apply(lambda x: \
            (1 if x['subjectivity'] > subjectivity_threshold else -1))
    df['sentiment'] = df['sentiment_details'].apply(lambda x: \
            (1 if x['polarity'] > sentiment_threshold else -1))
    del df['sentiment_details']

    return df


def binary_strong_negation(text, negation_words=None,
                           base_threshold=0.005):
    """
    Check if the text has a strong presence of negation words.
    
    Args:
        text (str): Text to analyze.
        negation_words (list): List of negation words to look for.
        base_threshold (float): Percentage of negation words in text to consider it strongly negated.
    
    Returns:
        int: 1 if strong negation is detected, else -1.
    """

    if not negation_words:
        negation_words = [
            'never',
            'no',
            'nothing',
            'nowhere',
            'noone',
            'none',
            'not',
            'hate',
            'worst',
            ]
    threshold = base_threshold * len(text)
    negation_count = sum(text.count(word) for word in negation_words)
    return (1 if negation_count > threshold else -1)


def binary_excessive_marks(text, base_threshold=0.005):
    """
    Check if the text has an excessive amount of question or exclamation marks.
    
    Args:
        text (str): Text to analyze.
        base_threshold (float): Percentage of total characters that are exclamation or question marks to be considered excessive.
    
    Returns:
        int: 1 if excessive marks are found, else -1.
    """

    threshold = base_threshold * len(text)
    return (1 if text.count('?') + text.count('!') > threshold else -1)


def binary_topic_diversity(text, stopwords_list=None,
                           base_threshold=0.6):
    """
    Evaluate the diversity of topics in the text based on the uniqueness of its words.
    
    Args:
        text (str): Text to analyze.
        stopwords_list (list): List of stopwords to ignore in the analysis.
        base_threshold (float): Percentage of unique words in text to consider it diverse.
    
    Returns:
        int: 1 if text shows topic diversity, else -1.
    """

    if stopwords_list is None:
        stopwords_list = stopwords.words('english')
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords_list]
    unique_words = set(words)
    threshold = base_threshold * len(words)
    return (1 if len(unique_words) > threshold else -1)


def preprocess_data(labels_file, tweets_file, output_file):
    """
    Process tweet data, merge with user labels, and overwrite the original dataset.
    
    Args:
        labels_file (str): Path to the CSV file containing user IDs and labels.
        tweets_file (str): Path to the CSV file containing user IDs and cleaned tweets.
        output_file (str): Path where the processed data should be saved, typically the same as labels_file.
    
    This function performs the following steps:
    1. Loads and processes the tweet and label data.
    2. Merges processed tweet features with user labels.
    3. Saves the merged DataFrame, potentially overwriting the original labels dataset.
    """

    df = load_data(labels_file, tweets_file)[['user_id', 'all_tweets']]
    df = binary_sentiment_subjectivity(df)
    df['binary_strong_negation'] = df['all_tweets'].apply(binary_strong_negation)
    df['binary_excessive_marks'] = df['all_tweets'].apply(binary_excessive_marks)
    df['binary_topic_diversity'] = df['all_tweets'].apply(binary_topic_diversity)
    df = df.drop(columns=['all_tweets'])

    label_data = pd.read_csv(labels_file)[['user_id', 'label']]
    processed_df = pd.merge(label_data, df, on='user_id', how='inner')
    processed_df.to_csv(output_file, index=False)

    return processed_df 