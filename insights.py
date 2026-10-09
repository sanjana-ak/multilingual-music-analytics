import pandas as pd

df = pd.read_csv("data/feature_engineered_music_data.csv")

print("DATASET LOADED SUCCESSFULLY!")

print("\nDataset Shape:")
print(df.shape)

print("\nLanguages:")
print(df["Language"].value_counts())



avg_duration = (
    df.groupby("Language")["Duration_sec"]
    .mean()
    .sort_values(ascending=False))

print("\nAVERAGE DURATION BY LANGUAGE:")
print(avg_duration.round(2))



print("\nDURATION CATEGORY DISTRIBUTION:")
print(df["Duration_Category"].value_counts())



print("\nAVERAGE NUMBER OF ARTISTS PER SONG:")
print(round(df["Artist_Count"].mean(), 2))

print("\nMAXIMUM NUMBER OF ARTISTS IN A SONG:")
print(df["Artist_Count"].max())



print("\nAVERAGE POPULARITY BY LANGUAGE:")

avg_popularity = (
    df.groupby("Language")["Popularity"]
    .mean()
    .sort_values(ascending=False))

print(avg_popularity.round(2))



print("\nTOP 5 SONGS BY DERIVED POPULARITY:")

top_songs = df.nlargest(
    5,
    "Popularity"
)[
    ["Song_Name", "Artist(s)", "Language", "Popularity"]]

print(top_songs.to_string(index=False))