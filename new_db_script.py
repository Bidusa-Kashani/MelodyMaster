import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import warnings
import re
np.random.seed(42)


df = pd.read_csv('./dataset.csv')
df = df.rename(columns={"songs": "lyrics", "song": "song_name"})
cleaned_df = df[df['lyrics'] != '[]']


for index, row in df.iterrows():
    if index == 0:
        continue
    song_url = row['url']
    reqs = requests.get(song_url,headers = {"Authorization": "Bearer bc2bfd5794465693e5e40098c30dafb5d3f01446"})
    soup = BeautifulSoup(reqs.text, 'html.parser')
    get_lyricist = soup.find("meta", itemprop="lyricist")
    lyricists = []
    if get_lyricist:
            lyricists = get_lyricist["content"].split(',')
            lyricists = list(map(str.strip, lyricists))
    print(lyricists)
    get_lyrics = soup.find("span",{'class' : 'artist_lyrics_text'})
    lyrics = " ".join(list(map(str.rstrip, get_lyrics.strings)))
    lyrics = ''.join(['' if 1456 <= ord(c) <= 1479 else c for c in lyrics])
