import pandas as pd
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Enable progress bar
tqdm.pandas()

# Load dataset
df = pd.read_csv("categorized_blogs_with_websites.csv")

# Ensure Date column is parsed
df['Published Date'] = pd.to_datetime(df['Published Date'], errors='coerce')  # Use correct column name

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Load BERT model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# VADER sentiment label
def get_sentiment_vader(text):
    score = vader_analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# VADER sentiment score
def get_sentiment_score_vader(text):
    return vader_analyzer.polarity_scores(text)['compound']

# BERT sentiment label
def get_sentiment_bert(text):
    try:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        output = model(**tokens)
        probs = softmax(output.logits, dim=1).detach().cpu().numpy()[0]
        labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        return labels[probs.argmax()]
    except:
        return "Unknown"

# Apply sentiment analysis with progress bar and time tracking
start = time.time()
df["Sentiment_VADER"] = df["Description"].progress_apply(get_sentiment_vader)
df["VADER_Score"] = df["Description"].progress_apply(get_sentiment_score_vader)
df["Sentiment_BERT"] = df["Description"].progress_apply(get_sentiment_bert)
print("Sentiment Analysis Completed in", round(time.time() - start, 2), "seconds")

# Save results
df.to_csv("blog_sentiment_analysis.csv", index=False)
print("Saved to blog_sentiment_analysis.csv")

# Plot VADER Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Sentiment_VADER", data=df, order=["Negative", "Neutral", "Positive"])
plt.title("VADER Sentiment Distribution")
plt.savefig("vader_sentiment_distribution.png", dpi=300)
plt.close()

# Plot BERT Sentiment Distribution
plt.figure(figsize=(8, 5))
order_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
sns.countplot(x="Sentiment_BERT", data=df, order=order_labels)
plt.title("BERT Sentiment Distribution")
plt.savefig("bert_sentiment_distribution.png", dpi=300)
plt.close()

# Time-based trend (if Date is present)
if "Published Date" in df.columns and df['Published Date'].notna().any():
    df["Month"] = df["Published Date"].dt.to_period("M")
    monthly_trend = df.groupby(["Month", "Sentiment_BERT"]).size().unstack().fillna(0)
    monthly_trend.plot(kind='line', figsize=(10, 6), title="Monthly Sentiment Trends (BERT)")
    plt.ylabel("Blog Count")
    plt.xlabel("Month")
    plt.savefig("monthly_sentiment_trends.png", dpi=300)
    plt.close()
    print("Trend graph saved as monthly_sentiment_trends.png")
else:
    print("Date column not available or invalid for trend analysis.")
