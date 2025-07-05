import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Load blog data
df = pd.read_csv("categorized_blogs_with_websites.csv")

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Load BERT model & tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Function to get VADER Sentiment
def get_sentiment_vader(text):
    score = vader_analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to get BERT Sentiment
def get_sentiment_bert(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    output = model(**tokens)
    probs = softmax(output.logits, dim=1).detach().numpy()[0]
    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    return labels[probs.argmax()]

# Apply sentiment analysis
df["Sentiment_VADER"] = df["Description"].apply(get_sentiment_vader)
df["Sentiment_BERT"] = df["Description"].apply(get_sentiment_bert)

# Save updated data
df.to_csv("blog_sentiment_analysis.csv", index=False)
print("Sentiment Analysis Completed and Saved!")
