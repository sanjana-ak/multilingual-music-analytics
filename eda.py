import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_music_data.csv")

print("==========================================")
print("       MUSIC DATASET - BASIC EDA")
print("==========================================")


print("\nTOTAL NUMBER OF SONGS:")
print(len(df))

print("\nSONGS BY LANGUAGE:")
print(df["Language"].value_counts())


print("\nDURATION STATISTICS (SECONDS):")
print(df["Duration_sec"].describe())


print("\nAVERAGE SONG DURATION BY LANGUAGE:")

average_duration = (
    df.groupby("Language")["Duration_sec"]
    .mean()
    .round(2)
    .sort_values(ascending=False))

print(average_duration)


print("\nPOPULARITY STATISTICS:")
print(df["Popularity"].describe())


print("\nAVERAGE POPULARITY BY LANGUAGE:")

average_popularity = (
    df.groupby("Language")["Popularity"]
    .mean()
    .round(2)
    .sort_values(ascending=False))

print(average_popularity)


print("\nTOP 10 SONGS BY POPULARITY:")

top_songs = (
    df.sort_values("Popularity", ascending=False)
    [["Song_Name", "Artist(s)", "Language", "Popularity"]]
    .head(10))

print(top_songs.to_string(index=False))



zero_popularity = (df["Popularity"] == 0).sum()
nonzero_popularity = (df["Popularity"] > 0).sum()

print("\nPOPULARITY DISTRIBUTION:")

print("Songs with popularity = 0:", zero_popularity)
print("Songs with popularity > 0:", nonzero_popularity)

print(
    "Percentage with popularity = 0:",
    round((zero_popularity / len(df)) * 100, 2),
    "%")



language_counts = df["Language"].value_counts()

plt.figure(figsize=(8, 5))

language_counts.plot(kind="bar")

plt.title("Number of Songs by Language")
plt.xlabel("Language")
plt.ylabel("Number of Songs")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()



avg_duration = df.groupby("Language")["Duration_sec"].mean()

plt.figure(figsize=(8, 5))

avg_duration.plot(kind="bar")

plt.title("Average Song Duration by Language")
plt.xlabel("Language")
plt.ylabel("Average Duration (Seconds)")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))

plt.hist(df["Duration_sec"], bins=15, edgecolor="black")

plt.title("Distribution of Song Durations")
plt.xlabel("Duration (Seconds)")
plt.ylabel("Number of Songs")

plt.tight_layout()
plt.show()



artists = (
    df["Artist(s)"]
    .str.split(",")
    .explode()
    .str.strip())

top_artists = artists.value_counts().head(10)

print("\nTOP 10 MOST FREQUENT ARTISTS:")
print(top_artists)


plt.figure(figsize=(9, 5))

top_artists.plot(kind="bar")

plt.title("Top 10 Most Frequent Artists")
plt.xlabel("Artist")
plt.ylabel("Number of Song Appearances")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))

plt.hist(
    df["Playlist_Rank"],
    bins=10,
    edgecolor="black")

plt.title("Distribution of Playlist Ranks")
plt.xlabel("Playlist Rank")
plt.ylabel("Number of Songs")

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))

plt.scatter(
    df["Playlist_Rank"],
    df["Duration_sec"],
    alpha=0.6)

plt.title("Song Duration vs Playlist Rank")
plt.xlabel("Playlist Rank")
plt.ylabel("Duration (Seconds)")

plt.tight_layout()
plt.show()

correlation = df["Playlist_Rank"].corr(df["Duration_sec"])

print("\nCORRELATION BETWEEN PLAYLIST RANK AND DURATION:")
print(round(correlation, 3))



popularity_status = pd.Series({
    "Popularity = 0": (df["Popularity"] == 0).sum(),
    "Popularity > 0": (df["Popularity"] > 0).sum()})

plt.figure(figsize=(7, 5))

popularity_status.plot(kind="bar")

plt.title("Distribution of Popularity Values")
plt.xlabel("Popularity Status")
plt.ylabel("Number of Songs")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()