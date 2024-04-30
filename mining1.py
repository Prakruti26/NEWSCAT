import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load your dataset
df = pd.read_csv('keywords_output.csv')

# Preprocessing
df['text'] = df['news_headline'] + ' ' + df['news_article'] + ' ' + df['keywords']  # Combine all text features
X = df['text']
y = df['news_category']

# Feature Engineering: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Model Training
model = MultinomialNB()  # Choose your model
model.fit(X_tfidf, y)

# Prediction function
def predict_category(news_headline, news_article):
    # Extract keyword from the headline (e.g., first word)
    keywords = news_headline.split()[0]
    new_text = news_headline + ' ' + news_article + ' ' + keywords
    new_text_tfidf = vectorizer.transform([new_text])
    predicted_category = model.predict(new_text_tfidf)
    return predicted_category[0]

# User Input
new_headline = input("Enter the new headline: ")
new_article = input("Enter the new article content: ")

# Predicted category
predicted_category = predict_category(new_headline, new_article)
print("Predicted category:", predicted_category)