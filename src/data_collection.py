# data_collection.py

import requests
import time
import pandas as pd
import re 
import json
import numpy as np
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from tqdm import tqdm
from rdflib import Graph
from difflib import SequenceMatcher
from collections import defaultdict

urls = [
"https://en.wikipedia.org/wiki/List_of_fiction_works_made_into_feature_films_(0-9,_A-C)",
"https://en.wikipedia.org/wiki/List_of_fiction_works_made_into_feature_films_(D-J)",
"https://en.wikipedia.org/wiki/List_of_fiction_works_made_into_feature_films_(K-R)",
"https://en.wikipedia.org/wiki/List_of_fiction_works_made_into_feature_films_(S-Z)",
"http://en.wikipedia.org/wiki/List_of_short_fiction_made_into_feature_films",
"https://en.wikipedia.org/wiki/List_of_children%27s_books_made_into_feature_films",
"https://en.wikipedia.org/wiki/List_of_films_based_on_comics"
]

TMDB_KEY = "4b7e821c38120d0216062a4e303aa73f"

def scrape_wikipedia_list(urls):
    rows = []
    for url in urls:
        r = requests.get(url,headers={"User-Agent":"DALAS-bot/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        tables = soup.find_all("table", {"class":"wikitable"})
        for table in tables:
            for tr in table.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all(["td"])]
                if len(tds) >= 2:
                    book = tds[0]
                    film = tds[1]
                    match_book = re.findall(r"\((?:19|20)\d{2}\)", book)
                    match_film = re.findall(r"\((?:19|20)\d{2}\)", film)
                    year_film = np.nan
                    year_book = np.nan
                    author = np.nan
                    if len(match_film) != 0 :
                        year_film = match_film[0].strip("(").strip(")")
                    if len(match_book) != 0:
                        year_book = match_book[0].strip("(").strip(")")
                    film = re.sub(r"\(.*?\)", "", film).strip()
                    film = re.sub(r"\[.*?\]", "", film).strip()
                    book = re.sub(r"\(.*?\)", "", book).strip()
                    book = re.sub(r"\[.*?\]", "", book).strip()
                    parts = book.rsplit(',', 1)  
                    if len(parts) == 2 :
                        book, author = parts
                    rows.append([book,author,film, year_book,year_film])
                if len(tds) == 1:                     
                    film = tds[0]
                    match = re.findall(r"\((?:19|20)\d{2}\)", film)
                    year = -1
                    if len(match) != 0 :
                        year = match[0].strip("(").strip(")")
                    film = re.sub(r"\(.*?\)", "", film).strip()
                    film = re.sub(r"\[.*?\]", "", film).strip()
                    rows.append([book, author ,film, year_book, year])
    df = pd.DataFrame(rows)
    return df

# Example usage:
# wiki_df = scrape_wikipedia_list("https://en.wikipedia.org/....")
# wiki_df.to_csv("dalas_project/data/raw/wikipedia_list.csv", index=False)

def tmdb_search_movie(title, year=None, api_key=TMDB_KEY,strict=False):
    params = {"api_key": api_key, "query": title, "page" : 1}
    if year:
        params["year"] = year
    r = requests.get("https://api.themoviedb.org/3/search/movie", params=params)
    r.raise_for_status()
    data = r.json()
    if data.get("results"):
        return data["results"][0]
    return None

def tmdb_get_movie_details(movie_id, api_key=TMDB_KEY):
    r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                     params={"api_key": api_key, "append_to_response": "credits"})
    r.raise_for_status()
    return r.json()

# --------------------------
# 4) Google Trends (pytrends)
# --------------------------

pytrends = TrendReq(hl='en-US', tz=0)  # tz=0 => UTC; align with Europe/Paris if needed
def get_google_trends_interest(term, start_date="2004-01-01", end_date="2025-12-31"):
    kw_list = [term]
    pytrends.build_payload(kw_list, timeframe=f'{start_date} {end_date}')
    df = pytrends.interest_over_time()
    if not df.empty:
        df = df.drop(columns=['isPartial'])
    return df


# =========================
# === books and authors dataset
# =========================

def normalize(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', re.sub(r'[\W_]+', ' ', text.lower())).strip()

def collect_df(movies_path, books_path, authors_path):
    # --- Adaptations --- #
    df = pd.read_csv(movies_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    df = df.reset_index(drop=True)
    df['adaptation_id'] = df.index + 1
    df['author_book'] = df['author_book'].fillna('').astype(str)
    df['book'] = df['book'].fillna('').astype(str)
    
    df['norm_author'] = df['author_book'].apply(normalize)
    df['norm_title'] = df['book'].apply(normalize)
    df['norm_film'] = df['film'].apply(normalize)
    # --- Books --- #
    df_books = pd.json_normalize(pd.read_json(books_path, lines=True)['_source'])
    if 'language' in df_books.columns:
        df_books = df_books[df_books['language'] == 'English']
    # only keep fiction book 
    if 'genres' in df_books.columns:
        df_books = df_books[df_books['genres'].apply(lambda x: 'Fiction' in x if isinstance(x, list) else False)]
    # remove useless features
    books_to_drop = ['link', 'thumbnail_url', 'createdAt', 'updatedAt','id', 'format', 'asin', 'titlecomplete','places',
                     'characters', 'edition_unique_id', 'isbn_10', 'isbn13', 'publisher', 'language', 
                     'num_pages', 'userRating']
    df_books.drop(columns=[b for b in books_to_drop if b in df_books.columns], inplace=True)
    
    # convert the date
    df_books['publish_date'] = pd.to_datetime(df_books['publish_date'], unit='ms', errors='coerce')
    # normalize names
    df_books['norm_title'] = df_books['title'].apply(normalize)
    df_books['norm_author'] = (
        df_books['author_id'].astype(str)
        .str.split('.').str[-1]
        .str.replace('_', ' ')
        .apply(normalize)
    )
    df_books['norm_series'] = df_books['series'].apply(lambda x: normalize(x[0]) if isinstance(x, list) and len(x) > 0 else '')
    # only keeping one line for each book
    df_books['publish_date'] = df_books.groupby(['norm_author', 'norm_title'])['publish_date'].transform('min')
    df_books['ratings_count'] = pd.to_numeric(df_books['ratings_count'], errors='coerce').fillna(0)
    df_books = df_books.sort_values(by='ratings_count', ascending=False)
    df_books = df_books.drop_duplicates(subset=['norm_title', 'norm_author'], keep='first')
    # find if a book was adaptated 
    book_adapt = (
        df.groupby(['norm_author', 'norm_title'])['adaptation_id']
        .apply(list).reset_index()
    )
    df_books = df_books.merge(book_adapt, on=['norm_author','norm_title'], how='left')
    
    # series handling (not working check duplicate id)
    series_matches = []
    for idx, book in df_books[df_books['norm_series'] != ''].iterrows():
        matching_films = df[df['norm_film'].str.contains(book['norm_series'], regex=False, na=False)]
        if not matching_films.empty:
            series_matches.append({
                'book_id': book['book_id'],
                'series_adaptation_id': matching_films['adaptation_id'].tolist()
            })

    if series_matches:
        series_df = pd.DataFrame(series_matches)
        df_books = df_books.merge(series_df, on='book_id', how='left')
        df_books['adaptation_ids'] = df_books.apply(
            lambda row: (row['adaptation_id'] if isinstance(row['adaptation_id'], list) else []) + 
                        (row['series_adaptation_id'] if isinstance(row['series_adaptation_id'], list) else []), 
            axis=1
        )
        df_books.drop(columns=['series_adaptation_id'], inplace=True)
    else:
        df_books['adaptation_ids'] = df_books['adaptation_id'].apply(lambda x: x if isinstance(x, list) else [])

    
    df_books['adapted'] = (df_books['adaptation_ids'].str.len() > 0).astype(int)
    df_books.drop(columns=['adaptation_id', 'norm_author', 'norm_title', 'norm_series'], inplace=True)
    # --- Authors --- #
    df_authors = pd.json_normalize(pd.read_json(authors_path, lines=True)['_source'])
    authors_to_drop = ['createdAt', 'author_image']
    df_authors.drop(columns=[c for c in authors_to_drop if c in df_authors.columns], inplace=True)
    df_authors['norm_author'] = df_authors['name'].apply(normalize)
    author_adapt = (
        df.groupby('norm_author')['adaptation_id']
        .apply(list).reset_index()
    )
    df_authors = df_authors.merge(author_adapt, on='norm_author', how='left')
    df_authors['adaptation_ids'] = df_authors['adaptation_id'].apply(lambda x: x if isinstance(x, list) else [])
    df_authors.drop(columns=['adaptation_id', 'norm_author'], inplace=True)
    
    df.drop(columns=['norm_author', 'norm_title', 'norm_film'], inplace=True)
    return df, df_books, df_authors