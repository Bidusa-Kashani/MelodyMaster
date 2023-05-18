import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import warnings
import ast
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

df = pd.read_csv('./dataset.csv')
df = df.rename(columns={"songs": "lyrics", "song": "song_name"})
#df = df[df['lyrics'] != '[]']
#df = df.reset_index()
#df.to_csv('./dataset.csv', index =False, encoding = 'utf-8-sig')

print(f"Dataset's shape: {df.shape}")
print(df.columns)

print(df.iloc[351])


def clean_niqqud(lyrics):
    return ''.join(['' if 1456 <= ord(c) <= 1479 else c for c in lyrics])


def line_format(list_of_strings):
    return '<s> ' + '<s> '.join('{} <\\s> '.format(line) for line in list_of_strings)


f = open("./lyricist.txt", "r", encoding='utf-8')
lyricist_list = ast.literal_eval(f.read())
f.close()
f = open("./lyrics.txt", "r", encoding='utf-8')
lyrics_list = ast.literal_eval(f.read())
f.close()
f = open("./artists.txt", "r", encoding='utf-8')
artists_list = ast.literal_eval(f.read())
f.close()

print(f"Starting at {len(lyrics_list)}")
for index in range(len(df)):
    if index < len(lyricist_list):
        continue
    if index % 25 == 0:
        print(index)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        f = open("./lyricist.txt", "w", encoding='utf-8')
        f.write(str(lyricist_list))
        f.close()
        f = open("./lyrics.txt", "w", encoding='utf-8')
        f.write(str(lyrics_list))
        f.close()
        f = open("./artists.txt", "w", encoding='utf-8')
        f.write(str(artists_list))
        f.close()
        print(len(lyricist_list), len(lyrics_list), len(artists_list))
        if len(lyricist_list) != len(lyrics_list) or len(lyricist_list) != len(artists_list) or len(
                lyricist_list) != index:
            print("OY VEYYYYYYYYYYYYYYYYYYYYYYYY")

    song_url = ""
    try:
        song_url = df.loc[index, 'url']
        identifier = df.loc[index, 'index']
        reqs = requests.get(song_url)
    except:
        print(song_url)
        lyricist_list[identifier] = None
        lyrics_list[identifier] = None
        artists_list[identifier] = None
        continue
    soup = BeautifulSoup(reqs.text, 'html.parser')
    get_lyricist = soup.find("meta", itemprop="lyricist")
    get_lyrics = soup.find("span", {'class': 'artist_lyrics_text'})
    get_artists = soup.find("a", {"class": "artist_singer_title"})
    if not get_lyricist or not get_lyrics:
        print(song_url)
        lyricist_list[identifier] = None
        lyrics_list[identifier] = None
        artists_list[identifier] = None
        continue
    lyricists = get_lyricist["content"].split(',')
    lyricists = list(map(str.strip, lyricists))
    lyricist_list[identifier] = lyricists
    if not get_artists:
        artists_list[identifier] = lyricists
    else:
        artist = get_artists.getText().strip()
        artists_list[identifier] = artist
    lyrics = line_format(list(map(str.rstrip, get_lyrics.strings)))
    lyrics = clean_niqqud(lyrics)
    lyrics_list[identifier] = lyrics


f = open("./lyricist.txt", "w", encoding='utf-8')
f.write(str(lyricist_list))
f.close()
f = open("./lyrics.txt", "w", encoding='utf-8')
f.write(str(lyrics_list))
f.close()
f = open("./artists.txt", "w", encoding='utf-8')
f.write(str(artists_list))
f.close()
