import pandas as pd

# ==========================================
# STEP 4: DATA TRANSFORMATION
#         & FEATURE ENGINEERING
# ==========================================

# Load cleaned dataset
df = pd.read_csv("data/cleaned_music_data.csv")

print("ORIGINAL DATASET SHAPE:")
print(df.shape)

print("\nORIGINAL COLUMNS:")
print(df.columns.tolist())


# ------------------------------------------
# 1. Create Duration Category
# ------------------------------------------

def categorize_duration(seconds):
    if seconds < 180:
        return "Short"
    elif seconds <= 240:
        return "Medium"
    else:
        return "Long"


df["Duration_Category"] = df["Duration_sec"].apply(
    categorize_duration
)

print("\nDURATION CATEGORIES:")
print(df["Duration_Category"].value_counts())

print("\nSAMPLE DURATION CATEGORY DATA:")
print(
    df[
        ["Song_Name", "Duration_sec", "Duration_Category"]
    ].head(10)
)


# ------------------------------------------
# 2. Create Artist Count Feature
# ------------------------------------------

df["Artist_Count"] = (
    df["Artist(s)"]
    .str.split(",")
    .apply(len)
)

print("\nARTIST COUNT STATISTICS:")
print(df["Artist_Count"].describe())

print("\nSAMPLE ARTIST COUNT DATA:")
print(
    df[
        ["Song_Name", "Artist(s)", "Artist_Count"]
    ].head(10)
)


# ------------------------------------------
# 3. Create Playlist Rank Category
# ------------------------------------------

def categorize_rank(rank):
    if rank <= 10:
        return "Top 10"
    elif rank <= 25:
        return "Rank 11-25"
    else:
        return "Rank 26-50"


df["Rank_Category"] = df["Playlist_Rank"].apply(
    categorize_rank
)

print("\nPLAYLIST RANK CATEGORIES:")
print(df["Rank_Category"].value_counts())

print("\nSAMPLE RANK CATEGORY DATA:")
print(
    df[
        ["Song_Name", "Playlist_Rank", "Rank_Category"]
    ].head(10)
)


# ------------------------------------------
# 4. One-Hot Encode Language
# ------------------------------------------

language_encoded = pd.get_dummies(
    df["Language"],
    prefix="Language",
    dtype=int
)

df = pd.concat(
    [df, language_encoded],
    axis=1
)

print("\nLANGUAGE ENCODED COLUMNS:")
print(language_encoded.columns.tolist())

print("\nSAMPLE LANGUAGE ENCODING:")
print(
    df[
        [
            "Language",
            "Language_English",
            "Language_Hindi",
            "Language_Malayalam",
            "Language_Tamil"
        ]
    ].head(10)
)


# ------------------------------------------
# 5. Create Derived Popularity Score
# ------------------------------------------

# Convert Playlist Rank (1-50) into a
# normalized Popularity score (0-100).
#
# Rank 1  = Popularity 100
# Rank 50 = Popularity 0

df["Popularity"] = (
    (50 - df["Playlist_Rank"]) / 49 * 100
).round(2)

print("\nDERIVED POPULARITY STATISTICS:")
print(df["Popularity"].describe())

print("\nSAMPLE POPULARITY SCORES:")
print(
    df[
        ["Song_Name", "Playlist_Rank", "Popularity"]
    ].head(10)
)


# ------------------------------------------
# 6. Save Feature-Engineered Dataset
# ------------------------------------------

df.to_csv(
    "data/feature_engineered_music_data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFEATURE-ENGINEERED DATASET SAVED SUCCESSFULLY!")

print("\nFINAL DATASET SHAPE:")
print(df.shape)

print("\nFINAL COLUMNS:")
print(df.columns.tolist())