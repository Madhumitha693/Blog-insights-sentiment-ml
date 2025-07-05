import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Load Dataset (Ensure the CSV file is in the correct location)
data = pd.read_csv("categorized_blogs_with_websites.csv")  # Update the path if needed

# 2️⃣ Ensure required columns exist
required_columns = ['Description', 'Category']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in dataset. Check CSV headers.")

# 3️⃣ Preprocessing - Handle missing values
data.dropna(subset=['Description', 'Category'], inplace=True)

# 4️⃣ Define Features (X) and Target Labels (y)
X = data['Description']  # Blog description as input text
y = data['Category']  # Blog category as classification label

# Convert text to numerical format using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Function to train and evaluate models
def train_and_evaluate(model, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)  # Train model
    end_time = time.time()
    
    y_pred = model.predict(X_test)  # Predict test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    train_time = end_time - start_time  # Compute training time
    
    print(f"{model_name}: Accuracy = {accuracy:.4f}, Training Time = {train_time:.4f} sec")
    return accuracy, train_time

# 5️⃣ Train and Compare Models
models = {
    "Naïve Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    results[name] = train_and_evaluate(model, name)

# 6️⃣ Display Results
df_results = pd.DataFrame(results, index=['Accuracy', 'Training Time']).T
print("\nModel Performance Comparison:")
print(df_results)

import matplotlib.pyplot as plt
import seaborn as sns

# Convert results to DataFrame
df_results = pd.DataFrame(results, index=['Accuracy', 'Training Time']).T

# Set Seaborn style
sns.set_style("whitegrid")

# 🔹 Accuracy Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x=df_results.index, y=df_results['Accuracy'], palette="viridis")
plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.show()

# 🔹 Training Time Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x=df_results.index, y=df_results['Training Time'], palette="coolwarm")
plt.title("Model Training Time Comparison", fontsize=14)
plt.ylabel("Training Time (seconds)")
plt.xlabel("Models")
plt.show()

