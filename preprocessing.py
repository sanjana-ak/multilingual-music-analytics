import pandas as pd

df = pd.read_excel(
    "data/raw_music_data.xlsx",
    sheet_name="Songs")


print("DATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)


print("\nMISSING VALUES:")
print(df.isnull().sum())


print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())


print("\nDUPLICATE SONG NAMES:")

duplicate_songs = df[
    df.duplicated(
        subset=["Song_Name"],
        keep=False)]

if duplicate_songs.empty:
    print("No duplicate song names found.")
else:
    print(
        duplicate_songs[
            ["Song_ID", "Song_Name", "Artist(s)", "Language"]])


df = df.drop_duplicates(
    subset=["Song_Name", "Artist(s)"],
    keep="first")

print("\nAFTER REMOVING DUPLICATE SONGS:")
print("Dataset shape:", df.shape)


print("\nNUMERICAL VALUE RANGES:")

print(
    "Duration:",
    df["Duration_sec"].min(),
    "to",
    df["Duration_sec"].max(),
    "seconds")

print(
    "Playlist Rank:",
    df["Playlist_Rank"].min(),
    "to",
    df["Playlist_Rank"].max())

print(
    "Popularity:",
    df["Popularity"].min(),
    "to",
    df["Popularity"].max())


print("\nLANGUAGES:")
print(df["Language"].value_counts())


df["Song_Name"] = df["Song_Name"].str.strip()
df["Artist(s)"] = df["Artist(s)"].str.strip()
df["Language"] = df["Language"].str.strip()


df = df.reset_index(drop=True)


df.to_csv(
    "data/cleaned_music_data.csv",
    index=False,
    encoding="utf-8-sig")

print("\nCLEANED DATASET SAVED SUCCESSFULLY!")
print("Final dataset shape:", df.shape)