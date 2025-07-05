import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("blog_sentiment_analysis.csv")

# Convert 'Published Date' to datetime format
df["Published Date"] = pd.to_datetime(df["Published Date"], errors='coerce')

# Drop rows with missing dates
df = df.dropna(subset=["Published Date"])

### ===== (1) SENTIMENT TREND ANALYSIS (Over Time) ===== ###
# Group by date and sentiment for trend visualization
trend_vader = df.groupby(["Published Date", "Sentiment_VADER"]).size().unstack().fillna(0)
trend_bert = df.groupby(["Published Date", "Sentiment_BERT"]).size().unstack().fillna(0)

# Plot VADER Sentiment Trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_vader, marker="o")
plt.title("Sentiment Trends Over Time (VADER)")
plt.xlabel("Date")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.grid(True)
plt.show()

# Plot BERT Sentiment Trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_bert, marker="o", palette="coolwarm")
plt.title("Sentiment Trends Over Time (BERT)")
plt.xlabel("Date")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.grid(True)
plt.show()

### ===== (2) SENTIMENT DISTRIBUTION COMPARISON ===== ###
# Count sentiment categories
vader_counts = df["Sentiment_VADER"].value_counts()
bert_counts = df["Sentiment_BERT"].value_counts()

# Plot VADER Sentiment Distribution
plt.figure(figsize=(12,5))
sns.barplot(x=vader_counts.index, y=vader_counts.values, palette="coolwarm")
plt.title("VADER Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Plot BERT Sentiment Distribution
plt.figure(figsize=(12,5))
sns.barplot(x=bert_counts.index, y=bert_counts.values, palette="magma")
plt.title("BERT Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

### ===== (3) AGREEMENT SCORE: VADER vs. BERT ===== ###
df["Agreement"] = df["Sentiment_VADER"] == df["Sentiment_BERT"]
agreement_score = df["Agreement"].mean() * 100
print(f"✅ VADER and BERT agree on {agreement_score:.2f}% of cases.")

### ===== (4) ENGAGEMENT CORRELATION (Optional) ===== ###
# If Engagement column exists, analyze engagement trends
if "Engagement" in df.columns:
    print("\n📊 Average Engagement by Sentiment (VADER):")
    print(df.groupby("Sentiment_VADER")["Engagement"].mean())

    print("\n📊 Average Engagement by Sentiment (BERT):")
    print(df.groupby("Sentiment_BERT")["Engagement"].mean())

### ===== (5) ACCURACY CHECK (If labeled data exists) ===== ###
# If the dataset contains an actual sentiment column for evaluation
if "Actual_Sentiment" in df.columns:
    print("\n🎯 VADER Performance:")
    print(classification_report(df["Actual_Sentiment"], df["Sentiment_VADER"]))

    print("\n🎯 BERT Performance:")
    print(classification_report(df["Actual_Sentiment"], df["Sentiment_BERT"]))
