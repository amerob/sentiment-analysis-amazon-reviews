# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

plt.style.use('ggplot')

# Read in data
df = pd.read_csv('Reviews.csv')
df = df.head(500)

# EDA
text = ' '.join(df['Text'].dropna())  # Concatenate all review text
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Review Text')
plt.show()

sns.countplot(x='Score', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Plot sentiment distribution
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution of Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Review Length Distribution
df['Review Length'] = df['Text'].apply(lambda x: len(str(x).split()))  # Word count
plt.figure(figsize=(10, 6))
sns.histplot(df['Review Length'], bins=30, kde=True, color='purple')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length (Number of Words)')
plt.ylabel('Frequency')
plt.show()

# Check for Missing Data
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
plt.figure(figsize=(10, 5))
sns.barplot(x=missing_data.index, y=missing_data.values, palette='viridis')
plt.title('Missing Data in Each Column')
plt.xlabel('Column')
plt.ylabel('Number of Missing Values')
plt.show()

# Basic NLTK
example = df['Text'][50]
nltk.download('punkt')
tokens = nltk.word_tokenize(example)
nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(tokens)
nltk.download('maxent_ne_chunker')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

# VADER Sentiment Scoring
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

sia.polarity_scores('I am so happy!')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)

# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

# Plot VADER results
ax = sns.barplot(data=vaders, x='Score', y='compound', hue="Score")
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0], palette='Blues')
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1], palette='Greens')
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2], palette='Reds')
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

# RoBERTa Pretrained Model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
}
print(scores_dict)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# Compare Scores between models
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='viridis',
             markers=['o', 's', 'D', 'P', '^'],
             height=2.5,
             diag_kind='hist',
             plot_kws={'alpha': 0.7})
plt.show()

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='tab10')
plt.show()

corr = results_df[['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of VADER and RoBERTa Sentiment Scores")
plt.show()

# Violin Plot for Sentiment Scores
plt.figure(figsize=(12, 8))

# VADER sentiment
plt.subplot(2, 2, 1)
sns.violinplot(x='Score', y='vader_neg', data=results_df, palette='tab10')
plt.title('VADER Negative Sentiment by Review Score')

plt.subplot(2, 2, 2)
sns.violinplot(x='Score', y='vader_neu', data=results_df, palette='tab10')
plt.title('VADER Neutral Sentiment by Review Score')

plt.subplot(2, 2, 3)
sns.violinplot(x='Score', y='vader_pos', data=results_df, palette='tab10')
plt.title('VADER Positive Sentiment by Review Score')

# RoBERTa sentiment
plt.subplot(2, 2, 4)
sns.violinplot(x='Score', y='roberta_neg', data=results_df, palette='tab10')
plt.title('RoBERTa Negative Sentiment by Review Score')

plt.tight_layout()
plt.show()

# KDE Plot for Sentiment Scores
plt.figure(figsize=(16, 8))

# VADER sentiment
plt.subplot(1, 2, 1)
sns.kdeplot(data=results_df, x='vader_neg', hue='Score', fill=True, common_norm=False, palette='tab10', alpha=0.6)
sns.kdeplot(data=results_df, x='vader_neu', hue='Score', fill=True, common_norm=False, palette='tab10', alpha=0.6)
sns.kdeplot(data=results_df, x='vader_pos', hue='Score', fill=True, common_norm=False, palette='tab10', alpha=0.6)
plt.title("VADER Sentiment Scores Distribution")
plt.xlabel('Sentiment Scores')
plt.ylabel('Density')

# RoBERTa sentiment
plt.subplot(1, 2, 2)
sns.kdeplot(data=results_df, x='roberta_neg', hue='Score', fill=True, common_norm=False, palette='tab10', alpha=0.6)
sns.kdeplot(data=results_df, x='roberta_neu', hue='Score', fill=True, common_norm=False, palette='tab10', alpha=0.6)
sns.kdeplot(data=results_df, x='roberta_pos', hue='Score', fill=True, common_norm=False, palette='tab10', alpha=0.6)
plt.title("RoBERTa Sentiment Scores Distribution")
plt.xlabel('Sentiment Scores')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# FacetGrid of Sentiment Scores by Review Score
g = sns.FacetGrid(results_df, col='Score', height=4, aspect=1.5)
g.map(sns.scatterplot, 'vader_pos', 'roberta_pos', alpha=0.6, color='blue')
g.set_axis_labels('VADER Positive Sentiment', 'RoBERTa Positive Sentiment')
g.set_titles('Review Score {col_name}')
plt.show()

# Review Examples
results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]

results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]

results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]

results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]

# Transformers Pipeline
sent_pipeline = pipeline("sentiment-analysis")
sent_pipeline('I hate sentiment analysis!')
sent_pipeline('It takes much time to do the sentiment analysis')
sent_pipeline('Hi!')
