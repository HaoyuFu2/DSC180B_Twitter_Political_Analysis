# DSC180B_Twitter_Political_Analysis

## Team Members: Yixuan Zhang, Haoyu Fu

## Content
The content includes:
1. Replication of the paper [AI-Augmented Surveys: Leveraging Large Language Models and Surveys for Opinion Prediction](https://arxiv.org/pdf/2305.09620.pdf) on a subset of the dataset used by the paper, using the GPT2 language model.
2. Code that predicts the political positions of Twitter users given their history tweets. This approach was inspired by the framework from the paper.

For detailed explanation of our code, please [check our website](https://haoyufu2.github.io/Twitter_Analysis_Website/) or read our [report]().

## Files and explanations

```
DSC180B_Twitter_Political_Analysis/
├── data/                               # Data directory containing datasets, generated files, etc.
│   ├── cleaned_tweets.csv                      # the cleaned original dataset
│   ├── labels_manual.csv                       # manually labelled dataset by our team members
│   ├── labels.csv                              # labelled dataset after sentiment analysis
│   ├── labels.parquet                          # intermediate files of sentiment analysis
│   └── weights.h5                              # intermediate files of weights from our training models
├── model_replicated/                   # Directory containing our replication on reference paper
│   ├── data/                                       # Data directory
│   │   ├── GSS_by_column.json                      # raw dataset 3
│   │   ├── gss.parquet                             # cleaned dataset 3
│   │   ├── missing_imputation.h5                   # cleaned dataset 3
│   │   ├── train_data.parquet                      # cleaned dataset 3
│   │   └── val_data.parquet                        # cleaned dataset 3
│   ├── demo_paper_replication.py               # main codes of the paper replication
│   ├── demo.ipynb                              # demo for the replication
│   ├── func_data_cleaning.py                   # data cleaning module of paper replication
│   └── gpt2_embeddings.pkl                     # intermediate files of embeddings
├── demo_prediction.ipynb                   # data cleaning module
├── demo_sentiment_and_clustering.ipynb     # tweets prediction module
├── func_LLM_pred.py                        # prediction module
├── func_user_sentiment_analysis.py         # sentiment analysis module
├── func_user_tweet_classifier.py           # clustering module
└── gpt2_embeddings.pkl                     # intermediate files of embeddings
```

## Installation and Reproduction Instructions
The following instructions show the steps to reproduce the results of our report. 
1. **Setting up Environment**: run the following command in **terminal/cmd** to install packages used in our project: ```pip install torch numpy pandas tensorflow tqdm tensorflow tensorflow_recommenders scikit-learn transformers pyarrow nltk textblob``` 

2. **Running the code**: Run all the cells of the following files in order:
    1. `demo_sentiment_and_clustering.ipynb`: Perform the clustering and sentiment analysis.
    2. `demo_prediction.ipynb`: Making predictions. The accuracy will be printed.

3. **(Optional) To replicate our reference paper**: Run all the cells from `model_replicated/demo.ipynb` in order. The results will be printed as the cell outputs.

4. **(Optional) To generate simulated user tweets**: Check our previous projec: [Optimizing Political Analysis Advanced Integration of LangChain with LLMs](https://github.com/GeorgeZhangDS/Optimizing-Political-Analysis-Advanced-Integration-of-LangChain-with-LLMs)
