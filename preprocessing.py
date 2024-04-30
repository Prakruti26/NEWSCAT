import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data (replace this with your dataset)
file_path = "inshort_news_data-1.csv\inshort_news_data-1.csv"

# Initialize NLTK resources
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Read data
df = pd.read_csv(file_path)

# Apply preprocessing to news headlines
df['cleaned_headlines'] = df['news_headline'].apply(preprocess_text)

# Combine cleaned article and headlines for TF-IDF
df['combined_text'] = df['news_article'] + ' ' + df['cleaned_headlines']

# Calculate TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get top keywords for each document
all_keywords = []
for i in range(len(df)):
    doc_keywords = []
    feature_index = tfidf_matrix[i, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
    sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: (x[1], x[0]), reverse=True)
    for feature_index, score in sorted_tfidf_scores[:5]:  # Top 5 keywords per document
        doc_keywords.append(feature_names[feature_index])
    all_keywords.append(doc_keywords)

# Add keywords to the DataFrame
df['keywords'] = all_keywords

# Save DataFrame to Excel
output_file = "keywords_output.xlsx"
df.to_excel(output_file, index=False)

print("Keywords extracted and saved to", output_file)
