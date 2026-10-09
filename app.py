import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import html
import base64
from pathlib import Path
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



st.set_page_config(
    page_title="Multilingual Music Analytics",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded")



@st.cache_data
def load_data():
    return pd.read_csv("data/feature_engineered_music_data.csv")


df = load_data()



GREEN = "#1DB954"
LIGHT_GREEN = "#55E989"
PURPLE = "#8B5CF6"
PINK = "#EC4899"
BLUE = "#38BDF8"

LANGUAGE_COLORS = {
    "Malayalam": GREEN,
    "Tamil": PURPLE,
    "Hindi": PINK,
    "English": BLUE}



st.markdown("""
<style>

/* =========================================================
   MAIN APP
========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(29,185,84,0.10),
            transparent 24%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(124,58,237,0.11),
            transparent 25%
        ),
        #08090c;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}


/* =========================================================
   HERO
========================================================= */

.hero {
    position: relative;
    overflow: hidden;

    padding: 48px 50px;
    margin-bottom: 28px;

    border-radius: 28px;

    background:
        linear-gradient(
            120deg,
            rgba(29,185,84,0.20),
            rgba(17,18,22,0.94) 42%,
            rgba(124,58,237,0.18)
        );

    border: 1px solid rgba(255,255,255,0.09);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.05);
}

.hero::after {
    content: "♪";

    position: absolute;

    right: 45px;
    top: -60px;

    font-size: 230px;
    font-weight: 800;

    color: rgba(255,255,255,0.035);

    transform: rotate(10deg);
}

.hero-badge {
    display: inline-block;

    padding: 7px 13px;

    border-radius: 999px;

    background: rgba(29,185,84,0.12);

    border: 1px solid rgba(29,185,84,0.30);

    color: #55e989;

    font-size: 12px;
    font-weight: 700;

    letter-spacing: 1.4px;

    margin-bottom: 18px;
}

.hero h1 {
    margin: 0;

    color: white;

    font-size: 48px;
    line-height: 1.05;

    font-weight: 800;

    letter-spacing: -2px;
}

.hero-gradient {
    background:
        linear-gradient(
            90deg,
            #ffffff,
            #9bf5bb,
            #bda7ff
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    max-width: 760px;

    margin-top: 18px;
    margin-bottom: 0;

    color: #a7adb7;

    font-size: 17px;
    line-height: 1.7;
}

.hero-languages {
    margin-top: 24px;

    color: #d9dce1;

    font-size: 13px;
    font-weight: 600;

    letter-spacing: 0.5px;
}


/* =========================================================
   KPI CARDS
========================================================= */

.kpi-card {
    min-height: 155px;

    padding: 23px 24px;

    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.075),
            rgba(255,255,255,0.025)
        );

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.20);

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);

    border-color:
        rgba(29,185,84,0.45);
}

.kpi-icon {
    font-size: 25px;

    margin-bottom: 13px;
}

.kpi-value {
    color: white;

    font-size: 35px;
    line-height: 1;

    font-weight: 800;

    letter-spacing: -1px;
}

.kpi-label {
    color: #858b96;

    margin-top: 10px;

    font-size: 12px;
    font-weight: 700;

    letter-spacing: 1.2px;
}


/* =========================================================
   SECTION HEADINGS
========================================================= */

.section-eyebrow {
    color: #1DB954;

    font-size: 11px;
    font-weight: 800;

    letter-spacing: 1.8px;

    margin-bottom: 5px;
}

.section-title {
    color: white;

    font-size: 28px;
    font-weight: 750;

    letter-spacing: -0.7px;

    margin-bottom: 6px;
}

.section-description {
    color: #818793;

    font-size: 14px;

    margin-bottom: 22px;
}


/* =========================================================
   SELECTION SUMMARY
========================================================= */

.selection-summary {
    margin-top: 18px;
    margin-bottom: 12px;

    padding: 16px 20px;

    border-radius: 16px;

    background:
        linear-gradient(
            90deg,
            rgba(29,185,84,0.08),
            rgba(139,92,246,0.06)
        );

    border:
        1px solid rgba(255,255,255,0.07);

    color: #969ca7;

    font-size: 13px;
}

.selection-summary strong {
    color: #f4f4f5;
}


/* =========================================================
   INSIGHT CARDS
========================================================= */

.insight-card {
    min-height: 155px;

    padding: 22px;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(29,185,84,0.09),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid rgba(29,185,84,0.16);
}

.insight-icon {
    font-size: 23px;

    margin-bottom: 13px;
}

.insight-title {
    color: white;

    font-size: 15px;
    font-weight: 700;

    margin-bottom: 7px;
}

.insight-text {
    color: #979da7;

    font-size: 13px;
    line-height: 1.55;
}

.insight-highlight {
    color: #55e989;

    font-weight: 700;
}


/* =========================================================
   LEADERBOARD
========================================================= */

.leaderboard {
    padding: 6px 0;
}

.leader-row {
    display: flex;

    align-items: center;

    margin-bottom: 10px;

    padding: 15px 17px;

    border-radius: 16px;

    background:
        linear-gradient(
            110deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.018)
        );

    border:
        1px solid rgba(255,255,255,0.07);

    transition: 0.2s ease;
}

.leader-row:hover {
    transform: translateX(4px);

    border-color:
        rgba(29,185,84,0.35);
}

.leader-rank {
    width: 50px;

    flex-shrink: 0;

    color: #55e989;

    font-size: 17px;
    font-weight: 800;
}

.leader-info {
    flex-grow: 1;

    min-width: 0;
}

.leader-name {
    color: #f7f7f8;

    font-size: 14px;
    font-weight: 700;

    white-space: nowrap;

    overflow: hidden;

    text-overflow: ellipsis;
}

.leader-meta {
    color: #747b86;

    margin-top: 3px;

    font-size: 11px;
}

.leader-score {
    margin-left: 15px;

    color: #f4f4f5;

    font-size: 16px;
    font-weight: 800;
}


/* =========================================================
   MEDIA / IMAGE CARDS
========================================================= */

.leader-avatar {
    width: 54px;
    height: 54px;
    flex-shrink: 0;
    margin-right: 13px;
    border-radius: 50%;
    object-fit: cover;
    object-position: center;
    border: 2px solid rgba(255,255,255,0.10);
    box-shadow: 0 5px 16px rgba(0,0,0,0.28);
}

.track-cover-small {
    width: 58px;
    height: 58px;
    flex-shrink: 0;
    margin-right: 13px;
    border-radius: 10px;
    object-fit: cover;
    object-position: center;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 5px 16px rgba(0,0,0,0.28);
}

.image-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(29,185,84,0.22), rgba(139,92,246,0.22));
    color: #d9dce1;
    font-size: 20px;
}

.top-track {
    display: flex;
    align-items: center;
    gap: 28px;
}

.top-track-cover {
    width: 190px;
    height: 190px;
    flex-shrink: 0;
    border-radius: 18px;
    object-fit: cover;
    object-position: center;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 18px 45px rgba(0,0,0,0.35);
}

.top-track-content {
    min-width: 0;
    flex-grow: 1;
}

@media (max-width: 800px) {
    .top-track {
        flex-direction: column;
        align-items: flex-start;
    }

    .top-track-cover {
        width: 150px;
        height: 150px;
    }
}


/* =========================================================
   TOP TRACK
========================================================= */

.top-track {
    margin-bottom: 18px;

    padding: 27px;

    border-radius: 22px;

    background:
        linear-gradient(
            125deg,
            rgba(29,185,84,0.16),
            rgba(139,92,246,0.08)
        );

    border:
        1px solid rgba(29,185,84,0.25);

    box-shadow:
        0 15px 40px rgba(0,0,0,0.20);
}

.top-track-label {
    color: #55e989;

    font-size: 11px;
    font-weight: 800;

    letter-spacing: 1.5px;
}

.top-track-name {
    color: white;

    margin-top: 8px;

    font-size: 24px;
    font-weight: 800;

    line-height: 1.25;
}

.top-track-artist {
    color: #9298a2;

    margin-top: 6px;

    font-size: 13px;
}

.top-track-score {
    color: white;

    margin-top: 20px;

    font-size: 36px;
    font-weight: 800;
}

.top-track-score span {
    color: #777e88;

    font-size: 13px;
    font-weight: 600;
}


/* =========================================================
   MODEL CARDS
========================================================= */

.model-card {
    min-height: 205px;

    padding: 24px;

    border-radius: 21px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.065),
            rgba(255,255,255,0.018)
        );

    border:
        1px solid rgba(255,255,255,0.08);
}

.model-badge {
    display: inline-block;

    margin-bottom: 16px;

    padding: 6px 10px;

    border-radius: 999px;

    color: #55e989;

    background:
        rgba(29,185,84,0.10);

    font-size: 10px;
    font-weight: 800;

    letter-spacing: 1px;
}

.model-name {
    color: white;

    margin-bottom: 18px;

    font-size: 21px;
    font-weight: 800;
}

.model-metrics {
    display: flex;

    gap: 24px;
}

.model-metric-value {
    color: #f7f7f8;

    font-size: 20px;
    font-weight: 800;
}

.model-metric-label {
    color: #777e88;

    margin-top: 3px;

    font-size: 10px;
    font-weight: 700;

    letter-spacing: 0.8px;
}


/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #111217 0%,
            #0d0e12 100%
        );

    border-right:
        1px solid rgba(255,255,255,0.06);
}

.sidebar-brand {
    padding:
        5px 4px 16px 4px;
}

.sidebar-logo {
    width: 46px;
    height: 46px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 14px;

    margin-bottom: 14px;

    font-size: 23px;

    background:
        linear-gradient(
            135deg,
            rgba(29,185,84,0.25),
            rgba(124,58,237,0.25)
        );

    border:
        1px solid rgba(255,255,255,0.08);
}

.sidebar-title {
    color: white;

    font-size: 20px;
    font-weight: 800;
}

.sidebar-subtitle {
    color: #777d87;

    font-size: 12px;
    line-height: 1.6;

    margin-top: 5px;
}

.sidebar-stats {
    margin-top: 12px;
    margin-bottom: 18px;

    padding: 13px 8px;

    border-radius: 13px;

    text-align: center;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.06);

    color: #8c929c;

    font-size: 11px;
    font-weight: 650;

    letter-spacing: 0.5px;
}


/* =========================================================
   TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;

    padding: 7px;

    border-radius: 16px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;

    padding-left: 18px;
    padding-right: 18px;

    border-radius: 11px;

    color: #8e949e;

    font-weight: 650;
}

.stTabs [aria-selected="true"] {
    color: white !important;

    background:
        linear-gradient(
            135deg,
            rgba(29,185,84,0.20),
            rgba(29,185,84,0.08)
        ) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #1DB954;
}


/* =========================================================
   DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {
    border-radius: 16px;

    overflow: hidden;

    border:
        1px solid rgba(255,255,255,0.07);
}


/* =========================================================
   STREAMLIT CLEANUP
========================================================= */

hr {
    border-color:
        rgba(255,255,255,0.07);
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* =========================================================
   FOOTER
========================================================= */

.custom-footer {
    text-align: center;

    margin-top: 60px;

    padding-top: 25px;

    border-top:
        1px solid rgba(255,255,255,0.06);

    color: #626872;

    font-size: 12px;

    line-height: 1.8;
}

.footer-title {
    color: #aeb3bc;

    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def style_chart(fig):

    fig.update_layout(
        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#A7ADB7"
        ),

        title=dict(
            font=dict(
                size=18,
                color="#F3F4F6"
            )
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False
    )

    return fig


def section_header(
    eyebrow,
    title,
    description
):

    content = (
        f'<div class="section-eyebrow">'
        f'{eyebrow}'
        f'</div>'

        f'<div class="section-title">'
        f'{title}'
        f'</div>'

        f'<div class="section-description">'
        f'{description}'
        f'</div>'
    )

    st.markdown(
        content,
        unsafe_allow_html=True
    )


# ============================================================
# LOCAL IMAGE HELPERS
# ============================================================

ARTIST_IMAGE_MAP = {
    "anirudh ravichander": "assets/artists/anirudh_ravichander.jpg",
    "sai abhyankkar": "assets/artists/sai_abhyankkar.jpg",
    "g. v. prakash": "assets/artists/gv_prakash.jpg",
    "g.v. prakash": "assets/artists/gv_prakash.jpg",
    "g v prakash": "assets/artists/gv_prakash.jpg",
    "g. v. prakash kumar": "assets/artists/gv_prakash.jpg",
    "g.v. prakash kumar": "assets/artists/gv_prakash.jpg",
    "g v prakash kumar": "assets/artists/gv_prakash.jpg",
    "arijit singh": "assets/artists/arijit_singh.jpg",
    "jakes bejoy": "assets/artists/jakes_bejoy.jpg",
    "arivu": "assets/artists/arivu.jpg",
    "shashwat sachdev": "assets/artists/shashwat_sachdev.jpg",
    "vivek": "assets/artists/vivek.jpg",
    "vivek velmurugan": "assets/artists/vivek.jpg",
    "irshad kamil": "assets/artists/irshad_kamil.jpg",
    "muthu": "assets/artists/muthu.jpg",
    "olivia rodrigo": "assets/artists/olivia_rodrigo.jpg",
    "david guetta": "assets/artists/david_guetta.jpg",
    "bruno mars": "assets/artists/bruno_mars.jpg",
    "ariana grande": "assets/artists/ariana_grande.jpg",
    "teddy swims": "assets/artists/teddy_swims.jpg",
    "bebe rexha": "assets/artists/bebe_rexha.jpg",
    "dababy": "assets/artists/dababy.jpg",
    "sienna spiro": "assets/artists/sienna_spiro.jpg",
    "harry styles": "assets/artists/harry_styles.jpg",
    "bella kay": "assets/artists/bella_kay.jpg",
    "hanan shah": "assets/artists/hanan_shaah.jpg",
    "hanan shaah": "assets/artists/hanan_shaah.jpg",
    "bibin ashok": "assets/artists/bibin_ashok.jpg",
    "vinayak sasikumar": "assets/artists/vinayak_sasikumar.jpg",
    "m.h.r": "assets/artists/m_h_r.jpg",
    "m.h.r.": "assets/artists/m_h_r.jpg",
    "m h r": "assets/artists/m_h_r.jpg",
    "joker390p": "assets/artists/joker390p.jpg",
    "arjn": "assets/artists/arjn.jpg",
    "sid sriram": "assets/artists/sid_sriram.jpg",
    "jasmine sandlas": "assets/artists/jasmine_sandlas.jpg",
    "sudhir yaduvanshi": "assets/artists/sudhir_yaduvanshi.jpg",
    "amit trivedi": "assets/artists/amit_trivedi.jpg",
    "anu malik": "assets/artists/anu_malik.jpg",
    "kumaar": "assets/artists/kumaar.jpg",
    "tanishk bagchi": "assets/artists/tanishk_bagchi.jpg",
    "a.r. rahman": "assets/artists/a_r_rahman.jpg",
    "a. r. rahman": "assets/artists/a_r_rahman.jpg",
    "a r rahman": "assets/artists/a_r_rahman.jpg",
    "a.r.rehman": "assets/artists/a_r_rahman.jpg",
    "a.r. rehman": "assets/artists/a_r_rahman.jpg",
    "sublahshini": "assets/artists/sublahshini.jpg",
    "subhlahshini": "assets/artists/sublahshini.jpg",
    "hiphop tamizha": "assets/artists/hiphop_tamizha.jpg",
    "santhosh narayanan": "assets/artists/santhosh_narayanan.jpg",
    "arunraja kamaraj": "assets/artists/arunraja_kamaraj.jpg",
    "sean roldan": "assets/artists/sean_roldan.jpg",
}

TRACK_COVER_MAP = {
    "kalyani": "assets/covers/kalyani.jpg",
    "phir se": "assets/covers/phir_se.jpg",
    "ala bolelo": "assets/covers/ala_bolelo.jpg",
    "tastes like summer": "assets/covers/tastes_like_summer.jpg",
    "amsham": "assets/covers/amsham.jpg",
    "alaakaa loova": "assets/covers/alaakaa_loova.jpg",
    "alaska loova": "assets/covers/alaakaa_loova.jpg",
    "the wedding song": "assets/covers/wedding_song.jpeg",
    "wedding song": "assets/covers/wedding_song.jpeg",
    "pattampoochi": "assets/covers/pattampoochi.jpg",
    "goindhamma": "assets/covers/goindhamma.jpg",
    "petal": "assets/covers/petal.jpg",
    "drop dead": "assets/covers/drop_dead.jpg",
    "runway": "assets/covers/runway.jpg",
    "less than a lover": "assets/covers/less_than_a_lover.jpg",
    "aari aari": "assets/covers/aari_aari.jpg",
    "dhurandhar": "assets/covers/dhurandhar_title.jpg",
    "dhurandhar title": "assets/covers/dhurandhar_title.jpg",
    "khat": "assets/covers/khat.jpg",
    "jaiye sajana": "assets/covers/jaiye_sajana.jpg",
    "asalayavale": "assets/covers/asalayavale.jpg",
    "ada bommale": "assets/covers/ada_bommale.jpg",
    "kinginichar": "assets/covers/kinginichar.jpg",
}


PREVIEW_AUDIO_MAP = {
    "kalyani": "assets/previews/kalyani.mp3",
    "phir se": "assets/previews/phir_se.mp3",
    "ala bolelo": "assets/previews/ale_bolelo.mp3",
    "tastes like summer": "assets/previews/tastes_like_summer.mp3",
    "amsham": "assets/previews/amsham.mp3",
    "alaakaa loova": "assets/previews/alaakaa_loova.mp3",
    "alaska loova": "assets/previews/alaakaa_loova.mp3",
    "the wedding song": "assets/previews/the_wedding_song.mp3",
    "wedding song": "assets/previews/the_wedding_song.mp3",
    "pattampoochi": "assets/previews/pattampoochi.mp3",
    "goindhamma": "assets/previews/goindhamma.mp3",
    "petal": "assets/previews/petal.mp3",
    "drop dead": "assets/previews/drop_dead.mp3",
    "runway": "assets/previews/runway.mp3",
    "less than a lover": "assets/previews/less_than_a_lover.mp3",
    "aari aari": "assets/previews/aari_aari.mp3",
    "dhurandhar": "assets/previews/dhurandhar.mp3",
    "khat": "assets/previews/khat.mp3",
    "jaiye sajana": "assets/previews/jaiye_sajana.mp3",
    "asalayavale": "assets/previews/asalayavale.mp3",
    "ada bommale": "assets/previews/ada_bommale.mp3",
    "kinginichar": "assets/previews/kinginichar.mp3",
}


def get_preview_audio(song_name):
    song_key = str(song_name).strip().lower()

    for title_key, path in PREVIEW_AUDIO_MAP.items():
        if title_key in song_key:
            return path

    return None


def image_to_data_uri(path):
    image_path = Path(path)

    if not image_path.exists():
        return None

    suffix = image_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def get_artist_image(artist_name):
    key = str(artist_name).strip().lower()
    path = ARTIST_IMAGE_MAP.get(key)
    return image_to_data_uri(path) if path else None


def get_track_cover(song_name):
    song_key = str(song_name).strip().lower()

    for title_key, path in TRACK_COVER_MAP.items():
        if title_key in song_key:
            return image_to_data_uri(path)

    return None


def artist_media_html(artist_name):
    image_uri = get_artist_image(artist_name)

    if image_uri:
        return (
            f'<img class="leader-avatar" '
            f'src="{image_uri}" '
            f'alt="{html.escape(str(artist_name))}">'
        )

    return '<div class="leader-avatar image-fallback">♪</div>'


def track_media_html(song_name, large=False):
    image_uri = get_track_cover(song_name)
    css_class = "top-track-cover" if large else "track-cover-small"

    if image_uri:
        return (
            f'<img class="{css_class}" '
            f'src="{image_uri}" '
            f'alt="{html.escape(str(song_name))}">'
        )

    return f'<div class="{css_class} image-fallback">♫</div>'


def spinning_record_player(audio_path, player_id="record_player"):
    record_path = Path("assets/record.png")
    audio_path = Path(audio_path)

    if not record_path.exists():
        st.caption("💿 Record image unavailable")
        return

    if not audio_path.exists():
        st.caption("🎧 Audio preview unavailable")
        return

    record_base64 = base64.b64encode(record_path.read_bytes()).decode("utf-8")
    audio_base64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")

    component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                overflow: hidden;
                font-family: Arial, sans-serif;
            }}

            .player {{
                display: flex;
                justify-content: flex-end;
                align-items: center;
                height: 150px;
                padding-right: 8px;
            }}

            .record-wrapper {{
                position: relative;
                width: 125px;
                height: 125px;
                cursor: pointer;
                user-select: none;
            }}

            .record {{
                width: 125px;
                height: 125px;
                object-fit: contain;
                transition: transform 0.15s ease;
                filter: drop-shadow(0 8px 18px rgba(0, 0, 0, 0.45));
            }}

            .record-wrapper:hover .record {{
                transform: scale(1.04);
            }}

            .record.spinning {{
                animation: spin 2.8s linear infinite;
            }}

            @keyframes spin {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}

            .status {{
                position: absolute;
                left: 50%;
                bottom: -4px;
                transform: translateX(-50%);
                padding: 4px 9px;
                border-radius: 999px;
                background: rgba(10, 10, 12, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.12);
                color: #ffffff;
                font-size: 10px;
                font-weight: 700;
                white-space: nowrap;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div class="player">
            <div class="record-wrapper" id="{player_id}_wrapper" title="Click to play / pause">
                <img id="{player_id}_record" class="record"
                     src="data:image/png;base64,{record_base64}"
                     alt="Play or pause preview">
                <div class="status" id="{player_id}_status">▶ PLAY</div>
            </div>

            <audio id="{player_id}_audio" preload="metadata">
                <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
            </audio>
        </div>

        <script>
            const wrapper = document.getElementById("{player_id}_wrapper");
            const record = document.getElementById("{player_id}_record");
            const audio = document.getElementById("{player_id}_audio");
            const status = document.getElementById("{player_id}_status");

            wrapper.addEventListener("click", async function() {{
                if (audio.paused) {{
                    try {{
                        await audio.play();
                        record.classList.add("spinning");
                        status.textContent = "❚❚ PAUSE";
                    }} catch (error) {{
                        status.textContent = "▶ PLAY";
                    }}
                }} else {{
                    audio.pause();
                    record.classList.remove("spinning");
                    status.textContent = "▶ PLAY";
                }}
            }});

            audio.addEventListener("ended", function() {{
                record.classList.remove("spinning");
                status.textContent = "▶ PLAY";
                audio.currentTime = 0;
            }});

            audio.addEventListener("pause", function() {{
                record.classList.remove("spinning");
                status.textContent = "▶ PLAY";
            }});
        </script>
    </body>
    </html>
    """

    components.html(component_html, height=155, scrolling=False)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    sidebar_html = (
        '<div class="sidebar-brand">'

        '<div class="sidebar-logo">'
        '🎧'
        '</div>'

        '<div class="sidebar-title">'
        'Music Analytics'
        '</div>'

        '<div class="sidebar-subtitle">'
        'Explore multilingual music trends through data.'
        '</div>'

        '</div>'
    )

    st.markdown(
        sidebar_html,
        unsafe_allow_html=True
    )


    sidebar_stats = (
        '<div class="sidebar-stats">'
        f'{len(df)} SONGS'
        '&nbsp; • &nbsp;'
        f'{df["Language"].nunique()} LANGUAGES'
        '&nbsp; • &nbsp;'
        '1 YEAR'
        '</div>'
    )

    st.markdown(
        sidebar_stats,
        unsafe_allow_html=True
    )


    st.markdown(
        "#### 🌍 Filter Languages"
    )


    selected_languages = st.multiselect(
        "Select languages",

        options=sorted(
            df["Language"].unique()
        ),

        default=sorted(
            df["Language"].unique()
        ),

        label_visibility="collapsed"
    )


    st.caption(
        "Charts update automatically based "
        "on your language selection."
    )


    st.markdown("---")


    st.markdown(
        """
**Languages**

🇮🇳 Malayalam  
🇮🇳 Tamil  
🇮🇳 Hindi  
🌎 English
        """
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    df["Language"].isin(
        selected_languages
    )
]


if filtered_df.empty:

    st.warning(
        "Select at least one language from the sidebar "
        "to explore the dashboard."
    )

    st.stop()


# ============================================================
# HERO
# ============================================================

hero_html = (
    '<div class="hero">'

    '<div class="hero-badge">'
    'DATA SCIENCE • MUSIC ANALYTICS'
    '</div>'

    '<h1>'
    'MULTILINGUAL<br>'

    '<span class="hero-gradient">'
    'MUSIC ANALYTICS'
    '</span>'

    '</h1>'

    '<p>'
    'Discover what the data sounds like. '
    'Explore music patterns, artists, duration trends '
    'and derived popularity across four languages '
    'through an interactive analytics experience.'
    '</p>'

    '<div class="hero-languages">'
    'MALAYALAM &nbsp; • &nbsp; '
    'TAMIL &nbsp; • &nbsp; '
    'HINDI &nbsp; • &nbsp; '
    'ENGLISH'
    '</div>'

    '</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True
)


# ============================================================
# KPI VALUES
# ============================================================

average_duration = (
    filtered_df[
        "Duration_sec"
    ].mean()
)

highest_popularity = (
    filtered_df[
        "Popularity"
    ].max()
)

number_languages = (
    filtered_df[
        "Language"
    ].nunique()
)


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3, k4 = st.columns(4)


with k1:

    st.markdown(
        (
            '<div class="kpi-card">'

            '<div class="kpi-icon">'
            '🎵'
            '</div>'

            f'<div class="kpi-value">'
            f'{len(filtered_df)}'
            f'</div>'

            '<div class="kpi-label">'
            'TOTAL SONGS'
            '</div>'

            '</div>'
        ),
        unsafe_allow_html=True
    )


with k2:

    st.markdown(
        (
            '<div class="kpi-card">'

            '<div class="kpi-icon">'
            '🌍'
            '</div>'

            f'<div class="kpi-value">'
            f'{number_languages}'
            f'</div>'

            '<div class="kpi-label">'
            'LANGUAGES'
            '</div>'

            '</div>'
        ),
        unsafe_allow_html=True
    )


with k3:

    st.markdown(
        (
            '<div class="kpi-card">'

            '<div class="kpi-icon">'
            '⏱️'
            '</div>'

            f'<div class="kpi-value">'
            f'{average_duration:.0f}s'
            f'</div>'

            '<div class="kpi-label">'
            'AVG DURATION'
            '</div>'

            '</div>'
        ),
        unsafe_allow_html=True
    )


with k4:

    st.markdown(
        (
            '<div class="kpi-card">'

            '<div class="kpi-icon">'
            '🔥'
            '</div>'

            f'<div class="kpi-value">'
            f'{highest_popularity:.0f}'
            f'</div>'

            '<div class="kpi-label">'
            'HIGHEST POPULARITY'
            '</div>'

            '</div>'
        ),
        unsafe_allow_html=True
    )


# ============================================================
# CURRENT SELECTION
# ============================================================

language_text = (
    " • ".join(
        selected_languages
    )
)


selection_summary = (
    '<div class="selection-summary">'

    'Currently exploring '

    f'<strong>'
    f'{len(filtered_df)} songs'
    f'</strong> '

    'across '

    f'<strong>'
    f'{number_languages} language(s)'
    f'</strong> '

    f'— {html.escape(language_text)}'

    '</div>'
)


st.markdown(
    selection_summary,
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

(
    tab_overview,
    tab_trends,
    tab_artists,
    tab_popularity,
    tab_ml,
    tab_dataset,
    tab_about
) = st.tabs(
    [
        "◉ Overview",
        "↗ Trends",
        "♬ Artists",
        "🔥 Popularity",
        "🤖 ML Lab",
        "▦ Dataset",
        "ⓘ About"
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with tab_overview:

    st.write("")


    section_header(
        "OVERVIEW",

        "Your dataset at a glance",

        "A quick look at language distribution and "
        "major patterns in the selected songs."
    )


    # --------------------------------------------------------
    # LANGUAGE DISTRIBUTION
    # --------------------------------------------------------

    language_counts = (
        filtered_df[
            "Language"
        ]
        .value_counts()
        .reset_index()
    )


    language_counts.columns = [
        "Language",
        "Songs"
    ]


    fig_language = px.bar(
        language_counts,

        x="Language",
        y="Songs",

        color="Language",

        color_discrete_map=
            LANGUAGE_COLORS,

        text="Songs",

        title="Songs by Language"
    )


    fig_language.update_traces(
        marker_line_width=0,

        textposition="outside"
    )


    fig_language.update_layout(
        showlegend=False
    )


    style_chart(
        fig_language
    )


    st.plotly_chart(
        fig_language,

        width="stretch"
    )


    # --------------------------------------------------------
    # INSIGHT VALUES
    # --------------------------------------------------------

    longest_language = (
        filtered_df
        .groupby(
            "Language"
        )[
            "Duration_sec"
        ]
        .mean()
        .idxmax()
    )


    common_duration = (
        filtered_df[
            "Duration_Category"
        ]
        .mode()[0]
    )


    avg_artist_count = (
        filtered_df[
            "Artist_Count"
        ]
        .mean()
    )


    # --------------------------------------------------------
    # INSIGHT CARDS
    # --------------------------------------------------------

    i1, i2, i3 = st.columns(3)


    with i1:

        st.markdown(
            (
                '<div class="insight-card">'

                '<div class="insight-icon">'
                '⏱️'
                '</div>'

                '<div class="insight-title">'
                'Longest Average Duration'
                '</div>'

                '<div class="insight-text">'

                '<span class="insight-highlight">'
                f'{html.escape(str(longest_language))}'
                '</span> '

                'currently has the highest '
                'average song duration.'

                '</div>'

                '</div>'
            ),
            unsafe_allow_html=True
        )


    with i2:

        st.markdown(
            (
                '<div class="insight-card">'

                '<div class="insight-icon">'
                '🎶'
                '</div>'

                '<div class="insight-title">'
                'Most Common Song Length'
                '</div>'

                '<div class="insight-text">'

                '<span class="insight-highlight">'
                f'{html.escape(str(common_duration))}'
                '</span> '

                'is the most frequent duration '
                'category in this selection.'

                '</div>'

                '</div>'
            ),
            unsafe_allow_html=True
        )


    with i3:

        st.markdown(
            (
                '<div class="insight-card">'

                '<div class="insight-icon">'
                '🎤'
                '</div>'

                '<div class="insight-title">'
                'Artist Collaboration'
                '</div>'

                '<div class="insight-text">'

                'Songs feature an average of '

                '<span class="insight-highlight">'
                f'{avg_artist_count:.2f} artists'
                '</span>.'

                '</div>'

                '</div>'
            ),
            unsafe_allow_html=True
        )


# ============================================================
# TRENDS TAB
# ============================================================

with tab_trends:

    st.write("")


    section_header(
        "TRENDS",

        "Duration patterns",

        "Compare average song lengths and explore "
        "the distribution of duration categories."
    )


    c1, c2 = st.columns(2)


    # --------------------------------------------------------
    # AVG DURATION
    # --------------------------------------------------------

    with c1:

        avg_duration = (
            filtered_df
            .groupby(
                "Language"
            )[
                "Duration_sec"
            ]
            .mean()
            .reset_index()
        )


        fig_duration = px.bar(
            avg_duration,

            x="Language",
            y="Duration_sec",

            color="Language",

            color_discrete_map=
                LANGUAGE_COLORS,

            text_auto=".1f",

            title=(
                "Average Duration "
                "by Language"
            )
        )


        fig_duration.update_layout(
            showlegend=False,

            yaxis_title="Seconds"
        )


        style_chart(
            fig_duration
        )


        st.plotly_chart(
            fig_duration,

            width="stretch"
        )


    # --------------------------------------------------------
    # DURATION CATEGORY
    # --------------------------------------------------------

    with c2:

        duration_counts = (
            filtered_df[
                "Duration_Category"
            ]
            .value_counts()
            .reset_index()
        )


        duration_counts.columns = [
            "Category",
            "Songs"
        ]


        fig_donut = px.pie(
            duration_counts,

            names="Category",
            values="Songs",

            hole=0.68,

            color_discrete_sequence=[
                GREEN,
                PURPLE,
                BLUE
            ],

            title=(
                "Song Duration "
                "Categories"
            )
        )


        fig_donut.update_traces(
            textposition="inside",

            textinfo="percent+label"
        )


        style_chart(
            fig_donut
        )


        st.plotly_chart(
            fig_donut,

            width="stretch"
        )


# ============================================================
# ARTISTS TAB
# ============================================================

with tab_artists:

    st.write("")


    section_header(
        "ARTISTS",

        "Artist leaderboard",

        "Discover which artists appear most frequently "
        "in your current selection."
    )


    artist_data = (
        filtered_df[
            "Artist(s)"
        ]
        .str.split(",")
        .explode()
        .str.strip()
    )


    artist_counts = (
        artist_data
        .value_counts()
        .head(10)
        .reset_index()
    )


    artist_counts.columns = [
        "Artist",
        "Songs"
    ]


    artist_left, artist_right = (
        st.columns(
            [1.05, 1.4]
        )
    )


    # --------------------------------------------------------
    # ARTIST LEADERBOARD — CLICKABLE POPOVERS
    # --------------------------------------------------------

    with artist_left:

        st.markdown("#### 🏆 Top Artists")
        st.caption("Click an artist to view details.")

        for index, row in artist_counts.iterrows():
            raw_artist = str(row["Artist"]).strip()
            song_count = int(row["Songs"])

            # All rows in the current filtered selection containing this artist
            artist_mask = (
                filtered_df["Artist(s)"]
                .fillna("")
                .str.split(",")
                .apply(lambda names: raw_artist in [name.strip() for name in names])
            )
            artist_rows = filtered_df[artist_mask].copy()
            artist_languages = ", ".join(sorted(artist_rows["Language"].dropna().unique()))
            artist_tracks = artist_rows.sort_values(["Playlist_Rank", "Song_Name"]).head(8)
            artist_image_path = ARTIST_IMAGE_MAP.get(raw_artist.lower())

            with st.popover(
                f"#{index + 1:02d}   {raw_artist}   ·   {song_count} song{'s' if song_count != 1 else ''}",
                width="stretch"
            ):
                pop_img, pop_info = st.columns([0.42, 1.0])

                with pop_img:
                    if artist_image_path and Path(artist_image_path).exists():
                        st.image(artist_image_path, width="stretch")
                    else:
                        st.markdown("### 🎤")

                with pop_info:
                    st.markdown(f"### {raw_artist}")
                    st.markdown(f"**Songs in current selection:** {song_count}")
                    st.markdown(f"**Language(s):** {artist_languages or '—'}")

                st.markdown("**Tracks in this dataset**")
                for _, track in artist_tracks.iterrows():
                    st.markdown(
                        f"• {track['Song_Name']} — {track['Language']} · "
                        f"Rank #{int(track['Playlist_Rank'])} · "
                        f"Popularity {track['Popularity']:.0f}"
                    )

    # --------------------------------------------------------
    # ARTIST CHART
    # --------------------------------------------------------

    with artist_right:

        fig_artist = px.bar(
            artist_counts,

            x="Songs",
            y="Artist",

            orientation="h",

            text="Songs",

            color="Songs",

            color_continuous_scale=[
                "#18251d",
                "#1DB954",
                "#55E989"
            ],

            title=(
                "Top 10 Most "
                "Frequent Artists"
            )
        )


        fig_artist.update_layout(
            yaxis={
                "categoryorder":
                    "total ascending"
            },

            coloraxis_showscale=False
        )


        style_chart(
            fig_artist
        )


        st.plotly_chart(
            fig_artist,

            width="stretch"
        )


# ============================================================
# POPULARITY TAB
# ============================================================

with tab_popularity:

    st.write("")


    section_header(
        "POPULARITY",

        "Top-ranked tracks",

        "Explore songs using the project's "
        "derived 0–100 popularity score."
    )


    st.caption(
        "ⓘ Popularity is derived from Playlist_Rank. "
        "It is not an external Spotify or YouTube metric."
    )


    top_songs = (
        filtered_df
        .sort_values(
            [
                "Popularity",
                "Playlist_Rank"
            ],

            ascending=[
                False,
                True
            ]
        )
        .head(10)
        .copy()
    )


    # --------------------------------------------------------
    # NUMBER ONE SONG
    # --------------------------------------------------------

    number_one = (
        top_songs.iloc[0]
    )


    number_one_name = (
        html.escape(
            str(
                number_one[
                    "Song_Name"
                ]
            )
        )
    )


    number_one_artist = (
        html.escape(
            str(
                number_one[
                    "Artist(s)"
                ]
            )
        )
    )


    number_one_language = (
        html.escape(
            str(
                number_one[
                    "Language"
                ]
            )
        )
    )


    top_track_html = (
        '<div class="top-track">'

        f'{track_media_html(number_one["Song_Name"], large=True)}'

        '<div class="top-track-content">'

        '<div class="top-track-label">'
        '★ HIGHEST-RANKED TRACK'
        '</div>'

        '<div class="top-track-name">'
        f'{number_one_name}'
        '</div>'

        '<div class="top-track-artist">'
        f'{number_one_artist}'
        ' • '
        f'{number_one_language}'
        ' • '
        f'Playlist Rank '
        f'#{int(number_one["Playlist_Rank"])}'
        '</div>'

        '<div class="top-track-score">'
        f'{number_one["Popularity"]:.0f}'

        '<span>'
        ' / 100 DERIVED SCORE'
        '</span>'

        '</div>'
        '</div>'
        '</div>'
    )


    st.markdown(
        top_track_html,

        unsafe_allow_html=True
    )

    with st.popover("🎧 View #1 track details"):
        detail_cover = None
        number_one_key = str(number_one["Song_Name"]).lower()
        for title_key, path in TRACK_COVER_MAP.items():
            if title_key in number_one_key:
                detail_cover = path
                break

        detail_cover_col, detail_info_col = st.columns([0.48, 1.0])
        with detail_cover_col:
            if detail_cover and Path(detail_cover).exists():
                st.image(detail_cover, width="stretch")
            else:
                st.markdown("### 🎵")
        with detail_info_col:
            st.markdown(f"### {number_one['Song_Name']}")
            st.markdown(f"**Artist(s):** {number_one['Artist(s)']}")
            st.markdown(f"**Language:** {number_one['Language']}")
            st.markdown(f"**Playlist Rank:** #{int(number_one['Playlist_Rank'])}")
            st.markdown(f"**Derived Popularity:** {number_one['Popularity']:.0f} / 100")
            st.markdown(f"**Duration:** {number_one['Duration']} ({int(number_one['Duration_sec'])} sec)")

        number_one_preview = get_preview_audio(number_one["Song_Name"])
        if number_one_preview:
            spinning_record_player(number_one_preview, "number_one_player")


    pop_left, pop_right = (
        st.columns(
            [1.05, 1.4]
        )
    )


    # --------------------------------------------------------
    # TOP 5 TRACKS — CLICKABLE POPOVERS
    # --------------------------------------------------------

    with pop_left:

        st.markdown("#### 🔥 Top Tracks")
        st.caption("Click a track to view details.")

        for position, (_, row) in enumerate(
            top_songs.head(5).iterrows(),
            start=1
        ):
            raw_song = str(row["Song_Name"])
            raw_artist = str(row["Artist(s)"])
            raw_language = str(row["Language"])

            cover_path = None
            song_key = raw_song.lower()
            for title_key, path in TRACK_COVER_MAP.items():
                if title_key in song_key:
                    cover_path = path
                    break

            with st.popover(
                f"#{position:02d}   {raw_song}   ·   {row['Popularity']:.0f}",
                width="stretch"
            ):
                cover_col, info_col = st.columns([0.48, 1.0])

                with cover_col:
                    if cover_path and Path(cover_path).exists():
                        st.image(cover_path, width="stretch")
                    else:
                        st.markdown("### 🎵")

                with info_col:
                    st.markdown(f"### {raw_song}")
                    st.markdown(f"**Artist(s):** {raw_artist}")
                    st.markdown(f"**Language:** {raw_language}")
                    st.markdown(f"**Playlist Rank:** #{int(row['Playlist_Rank'])}")
                    st.markdown(f"**Derived Popularity:** {row['Popularity']:.0f} / 100")
                    st.markdown(f"**Duration:** {row['Duration']} ({int(row['Duration_sec'])} sec)")
                    st.markdown(f"**Duration Category:** {row['Duration_Category']}")
                    st.markdown(f"**Artist Count:** {int(row['Artist_Count'])}")

                preview_path = get_preview_audio(raw_song)
                if preview_path:
                    spinning_record_player(preview_path, f"top_track_{position}_player")

    # --------------------------------------------------------
    # TOP 10 CHART
    # --------------------------------------------------------

    with pop_right:

        fig_popularity = px.bar(
            top_songs,

            x="Popularity",
            y="Song_Name",

            color="Language",

            color_discrete_map=
                LANGUAGE_COLORS,

            orientation="h",

            hover_data=[
                "Artist(s)",
                "Playlist_Rank"
            ],

            title=(
                "Top 10 by "
                "Derived Popularity"
            )
        )


        fig_popularity.update_layout(
            yaxis={
                "categoryorder":
                    "total ascending"
            }
        )


        style_chart(
            fig_popularity
        )


        st.plotly_chart(
            fig_popularity,

            width="stretch"
        )


# ============================================================
# ML LAB
# ============================================================

with tab_ml:

    st.write("")


    section_header(
        "MACHINE LEARNING",

        "Popularity Prediction Lab",

        "Comparing regression models using duration, "
        "artist count and language as input features."
    )


    # --------------------------------------------------------
    # MODEL CARDS
    # --------------------------------------------------------

    ml1, ml2 = (
        st.columns(2)
    )


    with ml1:

        linear_card = (
            '<div class="model-card">'

            '<div class="model-badge">'
            'MODEL 01'
            '</div>'

            '<div class="model-name">'
            'Linear Regression'
            '</div>'

            '<div class="model-metrics">'

            '<div>'
            '<div class="model-metric-value">'
            '20.73'
            '</div>'
            '<div class="model-metric-label">'
            'MAE'
            '</div>'
            '</div>'

            '<div>'
            '<div class="model-metric-value">'
            '25.18'
            '</div>'
            '<div class="model-metric-label">'
            'RMSE'
            '</div>'
            '</div>'

            '<div>'
            '<div class="model-metric-value">'
            '-0.0086'
            '</div>'
            '<div class="model-metric-label">'
            'R²'
            '</div>'
            '</div>'

            '</div>'

            '</div>'
        )


        st.markdown(
            linear_card,

            unsafe_allow_html=True
        )


    with ml2:

        rf_card = (
            '<div class="model-card">'

            '<div class="model-badge">'
            'MODEL 02'
            '</div>'

            '<div class="model-name">'
            'Random Forest'
            '</div>'

            '<div class="model-metrics">'

            '<div>'
            '<div class="model-metric-value">'
            '25.00'
            '</div>'
            '<div class="model-metric-label">'
            'MAE'
            '</div>'
            '</div>'

            '<div>'
            '<div class="model-metric-value">'
            '31.39'
            '</div>'
            '<div class="model-metric-label">'
            'RMSE'
            '</div>'
            '</div>'

            '<div>'
            '<div class="model-metric-value">'
            '-0.5675'
            '</div>'
            '<div class="model-metric-label">'
            'R²'
            '</div>'
            '</div>'

            '</div>'

            '</div>'
        )


        st.markdown(
            rf_card,

            unsafe_allow_html=True
        )


    st.write("")


    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    model_results = (
        pd.DataFrame(
            {
                "Model": [
                    "Linear Regression",
                    "Random Forest"
                ],

                "MAE": [
                    20.73,
                    25.00
                ],

                "RMSE": [
                    25.18,
                    31.39
                ],

                "R²": [
                    -0.0086,
                    -0.5675
                ]
            }
        )
    )


    m1, m2 = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # ERROR CHART
    # --------------------------------------------------------

    with m1:

        fig_error = px.bar(
            model_results,

            x="Model",

            y=[
                "MAE",
                "RMSE"
            ],

            barmode="group",

            color_discrete_sequence=[
                GREEN,
                PURPLE
            ],

            title=(
                "Model Error "
                "Comparison"
            )
        )


        style_chart(
            fig_error
        )


        st.plotly_chart(
            fig_error,

            width="stretch"
        )


    # --------------------------------------------------------
    # R2 CHART
    # --------------------------------------------------------

    with m2:

        fig_r2 = px.bar(
            model_results,

            x="Model",
            y="R²",

            text_auto=".4f",

            color="Model",

            color_discrete_sequence=[
                GREEN,
                PURPLE
            ],

            title=(
                "R² Score "
                "Comparison"
            )
        )


        fig_r2.update_layout(
            showlegend=False
        )


        style_chart(
            fig_r2
        )


        st.plotly_chart(
            fig_r2,

            width="stretch"
        )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    feature_columns = [
        "Duration_sec",
        "Artist_Count",
        "Language_English",
        "Language_Hindi",
        "Language_Malayalam",
        "Language_Tamil"
    ]


    X = df[
        feature_columns
    ]


    y = df[
        "Popularity"
    ]


    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,

        test_size=0.20,

        random_state=42
    )


    model = (
        LinearRegression()
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = (
        model.predict(
            X_test
        )
    )


    prediction_df = (
        pd.DataFrame(
            {
                "Actual":
                    y_test.values,

                "Predicted":
                    predictions
            }
        )
    )


    fig_prediction = (
        px.scatter(
            prediction_df,

            x="Actual",
            y="Predicted",

            title=(
                "Linear Regression — "
                "Actual vs Predicted"
            )
        )
    )


    fig_prediction.update_traces(
        marker=dict(
            size=9,

            color=GREEN,

            opacity=0.72,

            line=dict(
                width=1,

                color="#88F0AC"
            )
        )
    )


    fig_prediction.add_trace(
        go.Scatter(
            x=[0, 100],

            y=[0, 100],

            mode="lines",

            name="Ideal Prediction",

            line=dict(
                dash="dash",

                color=PURPLE
            )
        )
    )


    style_chart(
        fig_prediction
    )


    st.plotly_chart(
        fig_prediction,

        width="stretch"
    )


    # --------------------------------------------------------
    # MODEL INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        (
            '<div class="insight-card">'

            '<div class="insight-icon">'
            '🧠'
            '</div>'

            '<div class="insight-title">'
            'What did the models tell us?'
            '</div>'

            '<div class="insight-text">'

            'Linear Regression performed comparatively '
            'better than Random Forest, but both models '
            'showed weak predictive performance.'

            '<br><br>'

            'The results indicate that '

            '<span class="insight-highlight">'
            'duration, language and artist count alone '
            'are insufficient'
            '</span> '

            'to accurately predict the derived '
            'popularity score.'

            '</div>'

            '</div>'
        ),

        unsafe_allow_html=True
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

with tab_dataset:

    st.write("")


    section_header(
        "DATA EXPLORER",

        "Explore the music dataset",

        "Search, filter and sort the records used "
        "throughout the analytics platform."
    )


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_query = (
        st.text_input(
            "🔎 Search songs or artists",

            placeholder=(
                "Try: Anirudh, "
                "KALYANI, Pritam..."
            )
        )
    )


    explorer_df = (
        filtered_df.copy()
    )


    if search_query:

        search_mask = (
            explorer_df[
                "Song_Name"
            ]
            .astype(str)
            .str.contains(
                search_query,

                case=False,

                na=False,

                regex=False
            )

            |

            explorer_df[
                "Artist(s)"
            ]
            .astype(str)
            .str.contains(
                search_query,

                case=False,

                na=False,

                regex=False
            )
        )


        explorer_df = (
            explorer_df[
                search_mask
            ]
        )


    # --------------------------------------------------------
    # EXTRA FILTERS
    # --------------------------------------------------------

    filter1, filter2, filter3 = (
        st.columns(3)
    )


    with filter1:

        duration_options = (
            ["All"]

            +

            sorted(
                filtered_df[
                    "Duration_Category"
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )


        selected_duration = (
            st.selectbox(
                "⏱️ Duration Category",

                duration_options
            )
        )


    with filter2:

        sort_option = (
            st.selectbox(
                "↕️ Sort by",

                [
                    "Playlist Rank",
                    "Popularity",
                    "Duration",
                    "Song Name"
                ]
            )
        )


    with filter3:

        sort_direction = (
            st.selectbox(
                "📊 Order",

                [
                    "Ascending",
                    "Descending"
                ]
            )
        )


    # --------------------------------------------------------
    # DURATION FILTER
    # --------------------------------------------------------

    if selected_duration != "All":

        explorer_df = (
            explorer_df[
                explorer_df[
                    "Duration_Category"
                ]
                ==
                selected_duration
            ]
        )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    sort_column_map = {
        "Playlist Rank":
            "Playlist_Rank",

        "Popularity":
            "Popularity",

        "Duration":
            "Duration_sec",

        "Song Name":
            "Song_Name"
    }


    sort_column = (
        sort_column_map[
            sort_option
        ]
    )


    ascending = (
        sort_direction
        ==
        "Ascending"
    )


    explorer_df = (
        explorer_df
        .sort_values(
            sort_column,

            ascending=ascending
        )
    )


    # --------------------------------------------------------
    # RESULT COUNT
    # --------------------------------------------------------

    st.markdown(
        (
            '<div class="selection-summary">'

            'Showing '

            '<strong>'
            f'{len(explorer_df)} songs'
            '</strong> '

            'matching the current search '
            'and filters.'

            '</div>'
        ),

        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    preview_columns = [
        "Song_Name",
        "Artist(s)",
        "Language",
        "Duration",
        "Duration_Category",
        "Artist_Count",
        "Playlist_Rank",
        "Popularity"
    ]


    st.dataframe(
        explorer_df[
            preview_columns
        ],

        width="stretch",

        hide_index=True,

        height=560,

        column_config={

            "Song_Name":
                st.column_config.TextColumn(
                    "Song"
                ),

            "Artist(s)":
                st.column_config.TextColumn(
                    "Artist(s)"
                ),

            "Language":
                st.column_config.TextColumn(
                    "Language"
                ),

            "Duration":
                st.column_config.TextColumn(
                    "Duration"
                ),

            "Duration_Category":
                st.column_config.TextColumn(
                    "Category"
                ),

            "Artist_Count":
                st.column_config.NumberColumn(
                    "Artists"
                ),

            "Playlist_Rank":
                st.column_config.NumberColumn(
                    "Rank",
                    format="%d"
                ),

            "Popularity":
                st.column_config.ProgressColumn(
                    "Derived Popularity",

                    min_value=0,

                    max_value=100,

                    format="%.1f"
                )
        }
    )


    if explorer_df.empty:

        st.info(
            "No songs matched your current "
            "search and filter settings."
        )


# ============================================================
# ABOUT / METHODOLOGY
# ============================================================

with tab_about:

    st.write("")


    section_header(
        "ABOUT THE PROJECT",

        "How does this platform work?",

        "A quick look behind the dashboard — from raw "
        "music data to visualization and machine learning."
    )


    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        "### 🎧 Project Overview"
    )


    st.write(
        """
**Multilingual Music Analytics and Trend Visualization
Platform** is a data science project designed to explore
and compare music patterns across **Malayalam, Tamil,
Hindi and English** songs.

The project uses a curated multilingual music dataset
covering the project's one-year study period and applies
data preprocessing, exploratory data analysis, feature
engineering, machine learning and interactive
visualization.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Data Science Workflow"
    )


    workflow1, workflow2, workflow3 = (
        st.columns(3)
    )


    with workflow1:

        st.markdown(
            (
                '<div class="insight-card">'

                '<div class="insight-icon">'
                '📥'
                '</div>'

                '<div class="insight-title">'
                '01 — Data Preparation'
                '</div>'

                '<div class="insight-text">'
                'Music records were collected, '
                'inspected, cleaned and checked '
                'for missing values and duplicate '
                'songs.'
                '</div>'

                '</div>'
            ),

            unsafe_allow_html=True
        )


    with workflow2:

        st.markdown(
            (
                '<div class="insight-card">'

                '<div class="insight-icon">'
                '📊'
                '</div>'

                '<div class="insight-title">'
                '02 — Analysis'
                '</div>'

                '<div class="insight-text">'
                'EDA and feature engineering were '
                'used to investigate language, '
                'duration, artist and ranking '
                'patterns.'
                '</div>'

                '</div>'
            ),

            unsafe_allow_html=True
        )


    with workflow3:

        st.markdown(
            (
                '<div class="insight-card">'

                '<div class="insight-icon">'
                '🤖'
                '</div>'

                '<div class="insight-title">'
                '03 — Modeling & Reporting'
                '</div>'

                '<div class="insight-text">'
                'Regression models were evaluated '
                'and the final findings were '
                'presented through this interactive '
                'Streamlit dashboard.'
                '</div>'

                '</div>'
            ),

            unsafe_allow_html=True
        )


    st.write("")

    st.divider()


    # --------------------------------------------------------
    # POPULARITY METHODOLOGY
    # --------------------------------------------------------

    st.markdown(
        "### 🔥 How is Popularity Calculated?"
    )


    st.write(
        """
The dashboard uses a **derived popularity score**
rather than a Spotify, YouTube or other external
platform popularity metric.

The score is calculated from each song's
**Playlist Rank** and scaled from **0 to 100**.
        """
    )


    st.code(
        (
            "Popularity = "
            "((50 - Playlist_Rank) / 49) × 100"
        ),

        language=None
    )


    st.markdown(
        """
- **Rank 1 → Popularity 100**
- **Rank 50 → Popularity 0**
- Songs between these ranks receive proportionally scaled values.
        """
    )


    st.info(
        "The popularity score should be interpreted "
        "as a rank-derived analytical score, not as "
        "an actual stream count or platform popularity "
        "measurement."
    )


    st.divider()


    # --------------------------------------------------------
    # MACHINE LEARNING METHODOLOGY
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 Machine Learning Methodology"
    )


    st.write(
        """
The machine learning experiment investigates whether
basic song characteristics can predict the
**derived popularity score**.

The input features are:

**Duration • Artist Count • Language**

Two regression algorithms were evaluated:

**Linear Regression** and **Random Forest Regression**.
        """
    )


    ml_method1, ml_method2 = (
        st.columns(2)
    )


    with ml_method1:

        st.markdown(
            (
                '<div class="model-card">'

                '<div class="model-badge">'
                'TRAINING DESIGN'
                '</div>'

                '<div class="model-name">'
                '80 / 20 Split'
                '</div>'

                '<div class="insight-text">'
                '159 songs were used for training '
                'and 40 songs were used for testing '
                'with random_state = 42.'
                '</div>'

                '</div>'
            ),

            unsafe_allow_html=True
        )


    with ml_method2:

        st.markdown(
            (
                '<div class="model-card">'

                '<div class="model-badge">'
                'LEAKAGE PREVENTION'
                '</div>'

                '<div class="model-name">'
                'Playlist Rank Excluded'
                '</div>'

                '<div class="insight-text">'
                'Playlist_Rank and Rank_Category '
                'are not used as model inputs '
                'because the target Popularity '
                'score is derived directly from '
                'Playlist_Rank.'
                '</div>'

                '</div>'
            ),

            unsafe_allow_html=True
        )


    st.write("")

    st.divider()


    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.markdown(
        "### ⚠️ Model Limitations"
    )


    st.write(
        """
Both regression models showed **weak predictive
performance**. Linear Regression performed
comparatively better than Random Forest, but its
R² score was still approximately zero.

This indicates that **duration, language and artist
count alone do not explain the derived popularity
score well**.

A stronger real-world popularity prediction system
would require additional information such as
streaming activity, engagement, release timing,
artist-level metrics, audio characteristics or other
relevant signals.

The dataset also contains only **199 cleaned songs**,
so the machine learning results should be treated as
an exploratory experiment rather than a general model
of music popularity.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # DATASET SNAPSHOT
    # --------------------------------------------------------

    st.markdown(
        "### 📚 Dataset Snapshot"
    )


    about1, about2, about3, about4 = (
        st.columns(4)
    )


    about1.metric(
        "Songs",
        len(df)
    )


    about2.metric(
        "Languages",
        df[
            "Language"
        ].nunique()
    )


    about3.metric(
        "Features",
        len(
            df.columns
        )
    )


    about4.metric(
        "Study Period",
        "1 Year"
    )


    st.caption(
        "After preprocessing, one duplicate song "
        "record was removed from the original "
        "200-song dataset, leaving 199 records "
        "for analysis."
    )


# ============================================================
# FOOTER
# ============================================================

footer_html = (
    '<div class="custom-footer">'

    '<span class="footer-title">'
    'MULTILINGUAL MUSIC ANALYTICS & '
    'TREND VISUALIZATION PLATFORM'
    '</span>'

    '<br>'

    'Data Science Project'

    '&nbsp; • &nbsp;'

    'Malayalam'

    '&nbsp; • &nbsp;'

    'Tamil'

    '&nbsp; • &nbsp;'

    'Hindi'

    '&nbsp; • &nbsp;'

    'English'

    '</div>'
)


st.markdown(
    footer_html,
    unsafe_allow_html=True
)