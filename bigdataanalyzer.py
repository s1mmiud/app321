# ==============================
# TRENDING TOPICS NLP PROJECT
# ==============================

# Install required packages (run once)
!pip install pytrends pandas scikit-learn wordcloud matplotlib

from pytrends.request import TrendReq
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ------------------------------
# TASK 1: Fetch Trending Topics
# ------------------------------

# Connect to Google Trends
pytrends = TrendReq(hl='en-US', tz=360)

try:
    # Get trending searches (change country if needed: 'india', 'united_states', etc.)
    trending = pytrends.trending_searches(pn='united_states')
except Exception as e:
    print(f"Warning: Could not fetch live trending topics. Error: {e}")
    print("Using dummy data for demonstration.")
    # Create a dummy DataFrame if fetching fails
    trending = pd.DataFrame({
        'trend': [
            'artificial intelligence', 'machine learning', 'data science',
            'quantum computing', 'cybersecurity', 'cloud computing',
            'big data', 'blockchain technology', 'virtual reality', 'augmented reality'
        ]
    })

# Rename column
trending.columns = ['trend']

print("\nTrending Topics:")
print(trending.head())

# ------------------------------
# TASK 2: TF-IDF Calculation
# ------------------------------

documents = trending['trend'].tolist()

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

tfidf_scores = tfidf_df.sum().sort_values(ascending=False)

tfidf_table = tfidf_scores.reset_index()
tfidf_table.columns = ['word', 'tfidf_score']

print("\nTop TF-IDF Words:")
print(tfidf_table.head(10))

# Save TF-IDF table
tfidf_table.to_csv("tfidf_trending_words.csv", index=False)

# ------------------------------
# TASK 3: WordCloud Generation
# ------------------------------

word_freq = dict(zip(tfidf_table['word'], tfidf_table['tfidf_score']))

wordcloud = WordCloud(
    width=900,
    height=450,
    background_color='white'
).generate_from_frequencies(word_freq)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Save WordCloud image
wordcloud.to_file("trending_wordcloud.png")

print("\nFiles Generated:")
print("- tfidf_trending_words.csv")
print("- trending_wordcloud.png")
