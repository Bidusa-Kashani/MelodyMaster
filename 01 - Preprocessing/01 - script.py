import numpy as np
import pandas as pd
import ast
import warnings
from sklearn.model_selection import train_test_split
np.random.seed(42)


def niqqud_pred(lyrics_to_check):
    lyrics_to_check = ast.literal_eval(lyrics_to_check['lyrics'])
    short_words_count = sum(map(lambda x: len(x) < 4, lyrics_to_check))
    return short_words_count / len(lyrics_to_check) < 0.8


warnings.filterwarnings("ignore")
np.random.seed(42)
df = pd.read_csv('./dataset.csv')
df = df.rename(columns={"songs": "lyrics", "song": "song_name"})
cleaned_df = df[df['lyrics'] != '[]']
cleaned_df = cleaned_df[cleaned_df.apply(lambda x: niqqud_pred(x), axis=1)]
artists_order = list(cleaned_df.groupby('artist_key').count()['song_name'].sort_values().keys())
cleaned_df['artist_key'] = cleaned_df['artist_key'].astype('category')
cleaned_df['artist_key'] = cleaned_df['artist_key'].cat.set_categories(artists_order, ordered=True)
cleaned_df = cleaned_df.sort_values(by=['artist_key'],ascending=False)
unique_df = cleaned_df.drop_duplicates(subset=['lyrics'],keep='first')
unique_df['lyrics'] = unique_df['lyrics'].map(lambda l: ast.literal_eval(l))
k = 100
counts = unique_df['artist'].value_counts()
artists = counts[counts >= k].index
artists_to_remove = counts[counts < k].index
experimental = unique_df[unique_df['artist'].isin(artists_to_remove)]
experimental.to_csv('./experimental.csv', index=False, encoding='utf-8-sig')
unique_df = unique_df[unique_df['artist'].isin(artists)]
lyrics = unique_df['lyrics']
artists = unique_df['artist']
all_but_artists = unique_df[['lyrics','song_name','url','words count','unique words count']]
X_train, X_test, y_train, y_test = train_test_split(all_but_artists, artists, test_size=0.2, stratify=artists)
# Save to CSV
train_df = pd.DataFrame(np.c_[ X_train,y_train], columns = ['lyrics','song_name','url','words count','unique words count','artist'])
test_df = pd.DataFrame(np.c_[ X_test,y_test], columns = ['lyrics','song_name','url','words count','unique words count','artist'])
print(f'train shape: {train_df.shape}, test shape: {test_df.shape}')
train_df.to_csv('./train.csv', index=False,encoding = 'utf-8-sig')
test_df.to_csv('./test.csv', index=False,encoding = 'utf-8-sig')