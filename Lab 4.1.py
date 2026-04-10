import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Hàm tiền xử lý
def clean_text(t):
    if pd.isna(t): return ""
    t = t.lower()
    t = re.sub(r'[^a-zA-ZÀ-ỹ\s]', '', t)
    stop = ["và","nhưng","có","rất","hơi","chưa"]
    return " ".join([w for w in t.split() if w not in stop])

# Hàm TF-IDF
def make_tfidf(texts):
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    return X, vec

#Bài 1:
hotel = pd.read_csv("ITA105_Lab_4_Hotel_reviews.csv")
hotel["clean"] = hotel["review_text"].apply(clean_text)
X, v = make_tfidf(hotel["clean"])
print("Hotel TF-IDF:", X.shape)

#Bài 2:
match = pd.read_csv("ITA105_Lab_4_Match_comments.csv")
match["clean"] = match["comment_text"].apply(clean_text)
X, v = make_tfidf(match["clean"])
print("Match TF-IDF:", X.shape)

#Bài 3:
player = pd.read_csv("ITA105_Lab_4_Player_feedback.csv")
player["clean"] = player["feedback_text"].apply(clean_text)
X, v = make_tfidf(player["clean"])
print("Player TF-IDF:", X.shape)

#Bài 4:
album = pd.read_csv("ITA105_Lab_4_Album_reviews.csv")
album["clean"] = album["review_text"].apply(clean_text)
X, v = make_tfidf(album["clean"])
print("Album TF-IDF:", X.shape)