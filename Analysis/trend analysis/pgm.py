import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("blog_sentiment_analysis.csv")
# Sample DataFrame structure: df with 'date' and 'sentiment' columns
df['Published Date'] = pd.to_datetime(df['Published Date'])
df['Sentiment_VADER'] = df['Sentiment_VADER'].astype(str)  # Ensure sentiment is string

# Group by date and sentiment
trend_data = df.groupby([df['Published Date'].dt.to_period('M'), 'Sentiment_VADER']).size().unstack(fill_value=0)

# Plotting the sentiment trend over time
plt.figure(figsize=(12, 6))
trend_data.plot(kind='line', marker='o', linewidth=2)
plt.title('Monthly Sentiment Trend')
plt.xlabel('Month')
plt.ylabel('Number of Blogs')
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.show()
