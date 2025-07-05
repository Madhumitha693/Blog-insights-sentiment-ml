import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load sentiment data
df = pd.read_csv("blog_sentiment_analysis.csv")

# Convert date column
df["Published Date"] = pd.to_datetime(df["Published Date"], errors='coerce')
df = df.dropna(subset=["Published Date"])

# Count sentiment distributions
vader_counts = df["Sentiment_VADER"].value_counts(normalize=True) * 100
bert_counts = df["Sentiment_BERT"].value_counts(normalize=True) * 100

# Merge results for comparison
sentiment_comparison = pd.DataFrame({"VADER (%)": vader_counts, "BERT (%)": bert_counts}).fillna(0)
print("\nSentiment Distribution Comparison:\n", sentiment_comparison)

# Plot sentiment distribution
plt.figure(figsize=(10, 5))
sentiment_comparison.plot(kind="bar", colormap="coolwarm")
plt.title("Sentiment Distribution: VADER vs. BERT")
plt.xlabel("Sentiment Type")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45)
plt.legend(title="Model")
plt.show()

# Convert categorical sentiment to numeric for correlation
sentiment_mapping = {
    "Very Negative": -2, "Negative": -1, "Neutral": 0, "Positive": 1, "Very Positive": 2,
    "Negative": -1, "Positive": 1  # Adjust for VADER
}

df["Sentiment_VADER_Num"] = df["Sentiment_VADER"].map(sentiment_mapping)
df["Sentiment_BERT_Num"] = df["Sentiment_BERT"].map(sentiment_mapping)

# Drop NaN rows after mapping
df = df.dropna(subset=["Sentiment_VADER_Num", "Sentiment_BERT_Num"])

# Compute Pearson correlation
correlation, p_value = pearsonr(df["Sentiment_VADER_Num"], df["Sentiment_BERT_Num"])
print(f"\nCorrelation between VADER & BERT: {correlation:.3f} (p-value: {p_value:.5f})")

# Conclusion based on correlation
if correlation > 0.7:
    print("✅ VADER and BERT are highly correlated. Either can be used.")
elif correlation > 0.4:
    print("⚠️ VADER and BERT show moderate agreement. Check specific cases where they differ.")
else:
    print("❌ VADER and BERT have low agreement. Further validation is needed.")
