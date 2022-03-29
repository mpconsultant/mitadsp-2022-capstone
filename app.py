
from multiprocessing.dummy import Namespace
from urllib.request import Request
import pandas as pd
from flask import Flask, render_template,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
 
from scipy.sparse import coo_matrix
import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from scipy.stats import skew, norm, probplot
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


df = pd.read_csv('books.csv',error_bad_lines = False)
songs = pd.read_csv('songdata.csv')
songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
songs['text'] = songs['text'].str.replace(r'\n', '')

tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
lyrics_matrix = tfidf.fit_transform(songs['text'].values.astype('U'))
cosine_similarities = cosine_similarity(lyrics_matrix) 
similarities = {}
for i in range(len(cosine_similarities)):
    # Now we'll sort each element in cosine_similarities and get the indexes of the songs. 
    similar_indices = cosine_similarities[i].argsort()[:-50:-1] 
    # After that, we'll store in similarities each name of the 50 most similar songs.
    # Except the first one that is the same song.
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]


def print_message(song, recom_song):
        rec_items = len(recom_song)
        rec_list = []
        print(f'The {rec_items} recommended songs for {song} are:\n')
        init=f'The {rec_items} recommended songs for {song} are:\n'
        rec_list.append(init)
        rec_list.append('\n')
        for i in range(rec_items):
            
            st=str(i+1)  + ":  " + f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score"
             
            rec_list.append(st)
            rec_list.append('\n')
           
        return rec_list
    
def recommend_music(num_loc,num_songs):
       
         

        recommendation = {
                "song": songs['song'].iloc[int(num_loc)],
                 "number_songs": int(num_songs)
        }

        similarities = {}
        for i in range(len(cosine_similarities)):
                 
                 similar_indices = cosine_similarities[i].argsort()[:-50:-1] 
                
                 similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]

        
        song = recommendation['song']
        
        number_songs = recommendation['number_songs']
        
        recom_song = similarities[song][:number_songs]
       
        rec_list=print_message(song=song, recom_song=recom_song)
        stringo = ",".join(rec_list)
        stringnl =stringo.replace(',','\n')
        return stringnl

    
def recommend_books(bookid):
   
    df2 = df.copy()
    features = pd.concat([df2['average_rating'], 
                      df2['ratings_count']], axis=1)

    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)  
    model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features)
    dist, idlist = model.kneighbors(features)  
    book_list_name = []
    book_id = df2[df2['title'] == bookid].index
    book_id = book_id[0]
    i=1
    for newid in idlist[book_id]:
        book_list_name.append(" : " + str(i) +" : ")
        book_list_name.append(df2.loc[newid].title) 
        i+=1
    stringo = ",".join(book_list_name)
    stringnl =stringo.replace(',','')
    return stringnl

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/about',methods=['POST'])
def getvalue():
    
    number   = request.form['number']
    recommd = request.form['recommend']
    books   = request.form['books']
   

    
    if (books != " "):
       df=recommend_books(books)
       return render_template('index.html',prediction_text="{}".format(df))


    if (number  != " "): 
       if (recommd == "" ):
          df="Warning ====> Enter number of recommendations "
          return render_template('index.html',prediction_text="{}".format(df))

    if (recommd  != " "):
       if (number == "" ):
          df="Warning ====> Enter musicid  "
          return render_template('index.html',prediction_text="{}".format(df))
     
   
     
    
    df=recommend_music(number,recommd)
    return render_template('index.html',prediction_text="{}".format(df))

if __name__ == '__main__':
    app.run(debug=False)
