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


# ========================== 
# ====  extend the dataset with Film dataset
# ==========================

## --- Transform each relation into triplets
def get_relations(src):
    relations = []
    with open(src, 'r',encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            line = line.strip()
            parts = re.findall(r'<([^>]+)>|\t', line)
            parts = [p for p in parts if p and p != '\t']
            if len(parts) >= 3:
                entity, role, target = parts[0], parts[1], parts[2]
                if role.startswith('-'):
                    role = role[1:]
                relations.append({"entity": entity,"role": role, "target": target})
            elif len(parts) >= 2:
                valeur = re.search(r'"([^"]+)"', line)
                if valeur:
                    relations.append({"entity": parts[0],"role": parts[1],"target": valeur.group(1)})

    return relations

## --- get the film json info
def get_film_imdb(src):
    films = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            try:
                film = json.loads(line.strip())
                if film.get("Response") == 'True':
                    films.append(film)
            except:
                pass
    return films

# --- just to see all roles
def get_roles(relations):
    roles = set()
    for r in relations:
        roles.add(r['role'])
    return sorted(list(roles))


def extend_df(df, relations_film_dbpedia, relations_film_yago, dict_films_imdb):
    
    def normalize_title(title):
        return re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", re.sub(r"\(.*?\)", "", str(title if not pd.isna(title) else "")).strip().lower()))
    
    def extract_year(date_str):
        if pd.isna(date_str): return None
        try:
            year_match = re.search(r"\b(\d{4})\b", str(date_str))
            return int(year_match.group(1)) if year_match else None
        except:
            return None
        
    def extract_actors(actors_text, max_actors=3):
        if not actors_text: return []
        actors = [actor.strip() for actor in str(actors_text).split(",")]
        return actors[:max_actors]
    
    def convert_runtime(runtime_str):
        try:
            runtime = float(runtime_str)
            if runtime > 1000: 
                return str(round(runtime/ 60))  
            return str(round(runtime))
        except:
            return runtime_str
    
    # -- indexes for the three sources, too  long otherwise
    imdb_index = defaultdict(list)
    for film in dict_films_imdb:
        title_norm = normalize_title(film.get("Title", ""))
        imdb_index[title_norm].append(film)
    
    dbpedia_index = defaultdict(list)
    for r in relations_film_dbpedia:
        title_norm = normalize_title(r["entity"])
        dbpedia_index[title_norm].append(r)
    
    yago_index = defaultdict(list)
    for r in relations_film_yago:
        title_norm = normalize_title(r["entity"])
        yago_index[title_norm].append(r)
    
    def extend_row(row):
        film_title = row["film"]
        film_year = extract_year(row.get("date_film"))
        film_norm = normalize_title(film_title)
        
        if film_norm in imdb_index:
            films_imdb = imdb_index[film_norm]
            film_imdb = films_imdb[0]
            
            # -- check title and year (some films has the same name)
            if film_year is not None:
                for f in films_imdb:
                    f_year = extract_year(f.get("Year"))
                    if f_year == film_year:
                        film_imdb = f
                        break

            if pd.isna(row.get("director")) and film_imdb.get("Director"): 
                row["director"] = film_imdb.get("Director")
            if pd.isna(row.get("runtime")) and film_imdb.get("Runtime"): 
                row["runtime"] = film_imdb.get("Runtime")
            if pd.isna(row.get("genre_film")) and film_imdb.get("Genre"): 
                row["genre_film"] = film_imdb.get("Genre")
            if pd.isna(row.get("original_language")) and film_imdb.get("Language"): 
                row["original_language"] = film_imdb.get("Language")
            if pd.isna(row.get("awards")) and film_imdb.get("Awards"): 
                row["awards"] = film_imdb.get("Awards")
            if pd.isna(row.get("votes_film")) and film_imdb.get("imdbVotes"): 
                row["votes_film"] = film_imdb.get("imdbVotes")
            if pd.isna(row.get("vote_count_film")) and film_imdb.get("imdbVotes"): 
                row["vote_count_film"] = film_imdb.get("imdbVotes")
            if pd.isna(row.get("overview_film")) and film_imdb.get("Plot"): 
                row["overview_film"] = film_imdb.get("Plot")
            
            if pd.isna(row.get("actor1")) and film_imdb.get("Actors"):
                actors = extract_actors(film_imdb["Actors"])
                for i, actor in enumerate(actors):
                    row[f"actor{i+1}"] = actor
        
        if film_norm in dbpedia_index:
            actors_list = []
            for r in dbpedia_index[film_norm]:
                role, target = r["role"], r["target"]
                if role == "director" and pd.isna(row.get("director")): 
                    row["director"] = target
                elif role == "starring":
                    actors_list.append(target)
                elif role == "runtime" and pd.isna(row.get("runtime")): 
                    row["runtime"] = convert_runtime(target)
                elif role == "Work/runtime" and pd.isna(row.get("runtime")): 
                    row["runtime"] = target 
                elif role == "genre" and pd.isna(row.get("genre_film")): 
                    row["genre_film"] = target
                elif role == "language" and pd.isna(row.get("original_language")): 
                    row["original_language"] = target
                elif role == "budget" and pd.isna(row.get("budget_film")): 
                    row["budget_film"] = target
                elif role == "gross" and pd.isna(row.get("revenue_film")): 
                    row["revenue_film"] = target
                elif role == "award" and pd.isna(row.get("awards")): 
                    row["awards"] = target
            
            if pd.isna(row.get("actor1")) and actors_list:
                for i, actor in enumerate(actors_list[:3]):
                    row[f"actor{i+1}"] = actor
        
        if film_norm in yago_index:
            for r in yago_index[film_norm]:
                role, target = r["role"], r["target"]
                if role == "directed" and pd.isna(row.get("director")): 
                    row["director"] = target
                elif role == "actedIn" and pd.isna(row.get("actor1")): 
                    actors = extract_actors(target)
                    for i, actor in enumerate(actors):
                        if pd.isna(row.get(f"actor{i+1}")):
                            row[f"actor{i+1}"] = actor
                elif role == "hasDuration" and pd.isna(row.get("runtime")): row["runtime"] = convert_runtime(target)
        return row
    
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="extend film df"):
        ex_row = extend_row(row.copy())
        results.append(ex_row)
    
    ex_df = pd.DataFrame(results)
    return ex_df