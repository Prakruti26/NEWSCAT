import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('keywords_output.csv')

# Preprocessing

df['text'] = df['news_headline'] + ' ' + df['news_article'] + ' ' + df['keywords']
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
    new_text = news_headline + ' ' + news_article
    new_text_tfidf = vectorizer.transform([new_text])
    predicted_category = model.predict(new_text_tfidf)
    return predicted_category[0]

# Example usage
new_headline = "rbi's new upi rule chnage for ppis"
new_article = "unified payments interface and wallets like paytm,phonepe,and amazon have become integral to our daliy trans"
predicted_category = predict_category(new_headline, new_article)
print("Predicted category:", predicted_category)