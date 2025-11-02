# data_collection.py

import requests
import time
import pandas as pd
import re 
import numpy as np
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from tqdm import tqdm

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


