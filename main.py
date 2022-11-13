import pandas as pd
import neattext.functions as nfx
import streamlit as st
import pickle


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

# courses_list = pickle.load(open('udemy_courses.csv', 'rb'))

df  = pd.read_csv("udemy_courses.csv")
df.head()

df['course_title']
dir(nfx)
df['clean_course_title'] = df['course_title'].apply(nfx.remove_stopwords)
df['clean_course_title'] = df['clean_course_title'].apply(nfx.remove_special_characters)
df[['course_title','clean_course_title']]


count_vect = CountVectorizer()
cv_mat = count_vect.fit_transform(df['clean_course_title'])

cv_mat


cv_mat.todense()
df_cv_words = pd.DataFrame(cv_mat.todense(),columns=count_vect.get_feature_names())

df_cv_words.head()
cosine_sim_mat = cosine_similarity(cv_mat)
cosine_sim_mat

df.head()

course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()

course_indices['How To Maximize Your Profits Trading Options']

idx = course_indices['How To Maximize Your Profits Trading Options']
idx

scores = list(enumerate(cosine_sim_mat[idx]))
scores

sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
sorted_scores[1:]

selected_course_indices = [i[0] for i in sorted_scores[1:]]
selected_course_indices

selected_course_scores = [i[1] for i in sorted_scores[1:]]
recommended_result = df['course_title'].iloc[selected_course_indices]

rec_df = pd.DataFrame(recommended_result)
# rec_df.head()

rec_df['similarity_scores'] = selected_course_scores
rec_df

def recommend_course(title,num_of_rec=10):
    # ID for title
    idx = course_indices[title]
    # Course Indice
    # Search inside cosine_sim_mat
    scores = list(enumerate(cosine_sim_mat[idx]))
    # Scores
    # Sort Scores
    sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
    # Recomm
    selected_course_indices = [i[0] for i in sorted_scores[1:]]
    selected_course_scores = [i[1] for i in sorted_scores[1:]]
    result = df['course_title'].iloc[selected_course_indices]
    rec_df = pd.DataFrame(result)
    rec_df['similarity_scores'] = selected_course_scores
    return rec_df.head(num_of_rec) 

recommend_course('Learn and Build using Polymer',20)


st.markdown("<h2 style='text-align: center; color: blue;'>Coursera Course Recommendation System</h2>",
            unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Find similar courses from a dataset of over 3,000 courses from Coursera!</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Web App created Congle</h4>",
            unsafe_allow_html=True)

