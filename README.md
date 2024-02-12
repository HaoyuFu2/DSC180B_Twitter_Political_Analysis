# DSC180B_Twitter_Political_Analysis

## Team Members: Yixuan Zhang, Haoyu Fu

## Content
The progress so far includes:
1. Replication of the paper [AI-Augmented Surveys: Leveraging Large Language Models and Surveys for Opinion Prediction](https://arxiv.org/pdf/2305.09620.pdf) on a subset of the dataset used by the paper, using the GPT2 language model.
2. Model that predicts the candidate trends of the Twitter data using similiar methods as the paper.

## Files

```
DSC180B_Twitter_Political_Analysis/
├── data/                               # Data directory containing datasets, generated responses, etc.
│   ├── hashtag_joebiden.csv                    # raw dataset 1
│   ├── hashtag_donaldtrump.csv                 # raw dataset 2
│   ├── GSS_by_column.json                      # raw dataset 3
│   ├── cleaned_tweets_biden.csv                # cleaned dataset 1
│   ├── cleaned_tweets_trump.csv                # cleaned dataset 2
│   ├── gss.parquet                             # cleaned dataset 3
│   └── validation.parquet                      # validation dataset (manually labeled)
├── data_cleaning.py                            # data cleaning module
├── paper_replication.py                        # paper replication module
└── demo.ipynb          # Jupyter notebook for demonstrating the progress so far
```

## Installation and Reproduction Instructions
The following instructions show the steps to reproduce the results of our report. 
1. **Setting up Environment**: run the following command to install packages used in our project: ```pip install regex pandas openai seaborn matplotlib langchain``` 

2. **Downloading the raw data**: Download the [raw data](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets/). Make sure the two files `hashtag_donaldtrump.csv` and `hashtag_joebiden.csv` are located under `/data` directory.

3. **Data Cleaning**: Run all the cells from `data_cleaning.ipynb`. The results will be saved in `/data` directory.

4. **API keys**: An OpenAI API key is required to run our code. Go to the [OpenAI website](https://platform.openai.com/api-keys) to get a key. Open the `keys_config.py` file and replace the value of `openai.api_key` by the string of your API key.

5. **Generating responses**: Run all the cells from `demo.ipynb` to generate simulated responses from LLMs. The results will be processed and saved in `/data/in_context_responses.csv`. Note that the runtime can be greater than 10 minutes due to the large data size.

6. **Visualizing**: Run `visualization_overview.ipynb` and `visualizations_bias.ipynb` to check the final visualizations and results of our project report. Note that the runtime can be greater than 10 minutes due to the large data size.
