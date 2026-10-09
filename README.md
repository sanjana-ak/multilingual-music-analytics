# 🎧 Multilingual Music Analytics & Trend Visualization Platform

### Turning music data into meaningful insights. 🎶📊

A Python-based **Data Science and Machine Learning project** that explores music trends across four languages — **Malayalam, Tamil, Hindi, and English** — through data preprocessing, exploratory data analysis, feature engineering, predictive modeling, and an interactive Streamlit dashboard.

---

## 🌟 Project Overview

Music connects people across languages and cultures. But how do song characteristics, artist collaborations, and playlist rankings differ between languages?

This project explores these questions using a multilingual music dataset and transforms the results into an interactive, visually engaging analytics platform.

The platform combines **data analysis, machine learning, and interactive visualization** in a Spotify-inspired dashboard.

## ✨ Key Features

- 🎵 **Multilingual Music Analysis:** Explore Malayalam, Tamil, Hindi, and English songs.
- 📊 **Interactive Dashboard:** Visualize music data using Streamlit and Plotly.
- 🎤 **Artist Analytics:** Explore artist appearances and collaborations.
- 📈 **Exploratory Data Analysis:** Analyze song durations, playlist rankings, and language-wise distributions.
- 🧠 **Machine Learning:** Compare Linear Regression and Random Forest Regression models.
- 🎯 **Popularity Score Analysis:** Study a derived popularity score calculated from playlist rank.
- 🔎 **Dataset Explorer:** Filter and explore individual tracks.
- 💿 **Spotify-Inspired Interface:** A dark-themed dashboard featuring artist images, album artwork, and music-player styling.

## 🌍 Languages Covered

| Language | Songs |
|---|---:|
| Malayalam | 50 |
| Tamil | 49 |
| Hindi | 50 |
| English | 50 |
| **Total** | **199** |

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, OpenPyXL |
| Data Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Web Dashboard | Streamlit |
| Development Tools | VS Code, Git, GitHub |

## 🔄 Project Workflow

The project follows seven stages:

1. **Data Collection** — Organize multilingual music records into a structured dataset.
2. **Data Preprocessing** — Clean the dataset, remove duplicates, and validate values.
3. **Exploratory Data Analysis** — Examine song durations, rankings, and language-wise trends.
4. **Feature Engineering** — Create useful variables such as artist count, duration category, and language indicators.
5. **Data Modeling** — Train and evaluate regression models.
6. **Interpretation & Insights** — Analyze the results and identify patterns and limitations.
7. **Visualization & Reporting** — Present the findings through an interactive Streamlit dashboard.

## 🧠 Machine Learning

Two regression algorithms were evaluated:

- Linear Regression
- Random Forest Regression

The models use song duration, artist count, and language indicators to predict a **rank-derived popularity score**.

### Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 20.73 | 25.18 | -0.009 |
| Random Forest Regression | 25.00 | 31.39 | -0.568 |

**Interpretation:** Linear Regression performed better than Random Forest on the test set, but both models showed weak predictive performance. This suggests that the available song characteristics alone are insufficient to reliably predict the derived popularity score.

**Important:** The popularity score is calculated from playlist rank and is not an official Spotify popularity metric.

## 📁 Project Structure

```text
spotify_ds/
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── artists/
│   ├── covers/
│   └── record.png
├── data/
│   ├── raw_music_data.xlsx
│   ├── cleaned_music_data.csv
│   └── feature_engineered_music_data.csv
├── app.py
├── preprocessing.py
├── eda.py
├── feature_engineering.py
├── model.py
├── insights.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sanjana-ak/multilingual-music-analytics.git
cd multilingual-music-analytics
```

### 2. Install the dependencies

Ensure Python is installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Streamlit will display a local URL where you can explore the dashboard.

### 4. Run the data science pipeline (optional)

To regenerate the analysis outputs, run the scripts in this order:

```bash
python preprocessing.py
python eda.py
python feature_engineering.py
python model.py
python insights.py
```

## 📸 Dashboard Preview

Dashboard screenshots will be added here.

## 🔍 Key Findings

- The cleaned dataset contains **199 songs** across four languages.
- Average song duration is approximately **215.43 seconds**.
- Song duration has almost no linear relationship with playlist rank in this dataset.
- Language-wise average derived popularity scores are similar, partly because of the dataset's collection and scoring design.
- The evaluated regression models have limited predictive power, highlighting the need for richer musical and engagement features.

## 🎶 Music Preview Note

The local version of the dashboard supports music-preview playback.

Audio preview files are **not included in this public repository** due to music copyright and redistribution considerations. The dashboard's data analysis and visualization features can be explored without these files.

## 🔮 Future Enhancements

- Integration with authorized music-data APIs
- Additional audio and engagement features
- Improved popularity prediction models
- Larger datasets spanning more languages and time periods
- Automated data updates
- Deployment of the interactive dashboard

---

### 💚 Made with Python, data, and a love for music.

*An academic Data Science project exploring multilingual music trends through analytics, machine learning, and interactive visualization.*