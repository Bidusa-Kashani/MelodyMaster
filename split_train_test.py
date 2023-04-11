import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

df = pd.read_csv('kaggle.csv')
cleaned_df = df[df['songs'] != '[]']
idx = np.random.permutation(np.arange(len(cleaned_df)))
cleaned_df = cleaned_df.iloc[idx].drop_duplicates(subset=['songs'])

# TODO: set k value
k = 50
counts = cleaned_df['artist'].value_counts()
artists_to_keep = counts[counts >= k].index
cleaned_df = cleaned_df[cleaned_df['artist'].isin(artists_to_keep)]

lyrics = cleaned_df['songs'].to_numpy()
artists = cleaned_df['artist'].to_numpy() # or is it 'artist_key'?

X_train, X_test, y_train, y_test = train_test_split(lyrics, artists, test_size=0.2, stratify=artists)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

# Save to CSV
train_df = pd.DataFrame({'lyrics': X_train, 'artist': y_train})
val_df = pd.DataFrame({'lyrics': X_val, 'artist': y_val})
test_df = pd.DataFrame({'lyrics': X_test, 'artist': y_test})

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)
