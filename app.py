import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem


# Page config
st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎵",
    layout="centered"
)

# Custom CSS for Spotify-like dark theme accents
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .song-card {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1DB954;
    }
    .now-playing {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1ED760;
        box-shadow: 0 4px 6px rgba(30, 215, 96, 0.1);
    }
    .song-title {
        color: #ffffff;
        font-size: 1.1em;
        font-weight: 600;
        margin: 0;
    }
    .artist-name {
        color: #b3b3b3;
        font-size: 0.95em;
        margin: 0;
    }
    .rec-number {
        color: #1DB954;
        font-weight: 700;
        font-size: 1.2em;
    }
    .header-container {
        text-align: center;
        padding: 20px 0;
    }
    .method-badge {
        background-color: #1DB954;
        color: #000000;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    songs_data = pd.read_csv("data/cleaned_data.csv")
    transformed_data = load_npz("data/transformed_data.npz")
    track_ids = load("data/track_ids.npy", allow_pickle=True)
    filtered_data = pd.read_csv("data/collab_filtered_data.csv")
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")
    return songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid_data

songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid_data = load_data()


# Header
st.markdown("<div class='header-container'>", unsafe_allow_html=True)
st.markdown("# 🎵 Spotify Song Recommender")
st.markdown("*Discover your next favorite song using AI-powered hybrid recommendations*")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Input section
col1, col2 = st.columns(2)

with col1:
    song_name = st.text_input("🎶 Song Name", placeholder="e.g., Mr. Brightside")

with col2:
    artist_name = st.text_input("🎤 Artist Name", placeholder="e.g., The Killers")

# Lowercase inputs
song_name = song_name.lower().strip()
artist_name = artist_name.lower().strip()

# Settings row
col_k, col_method = st.columns(2)

with col_k:
    k = st.selectbox("📊 Number of Recommendations", [5, 10, 15, 20], index=1)

# Determine filtering type
if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():
    filtering_type = "Hybrid Recommender System"
else:
    filtering_type = "Content-Based Filtering"

with col_method:
    st.text_input("🧠 Method", value=filtering_type, disabled=True)

# Diversity slider for hybrid
if filtering_type == "Hybrid Recommender System":
    st.markdown("#### ⚖️ Recommendation Balance")
    col_slider, col_info = st.columns([3, 1])
    
    with col_slider:
        diversity = st.slider(
            "Drag to adjust: Left = Personalized, Right = Diverse",
            min_value=1,
            max_value=9,
            value=5,
            step=1
        )
    
    with col_info:
        content_based_weight = 1 - (diversity / 10)
        st.metric("Content Weight", f"{content_based_weight:.1f}")
        st.metric("Collab Weight", f"{diversity/10:.1f}")

st.markdown("---")

# Recommend button
recommend_clicked = st.button("🚀 Get Recommendations", use_container_width=True, type="primary")

if recommend_clicked:
    if not song_name or not artist_name:
        st.warning("⚠️ Please enter both a song name and artist name.")
    elif filtering_type == "Content-Based Filtering":
        if ((songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)).any():
            
            with st.spinner("Finding similar songs..."):
                recommendations = content_recommendation(
                    song_name=song_name,
                    artist_name=artist_name,
                    songs_data=songs_data,
                    transformed_data=transformed_data,
                    k=k
                )
            
            st.markdown(f"### Recommendations for **{song_name.title()}** by **{artist_name.title()}**")
            st.markdown(f"<span class='method-badge'>Content-Based</span>", unsafe_allow_html=True)
            st.markdown("")
            
            for ind, recommendation in recommendations.iterrows():
                rec_song = recommendation['name'].title()
                rec_artist = recommendation['artist'].title()
                preview_url = recommendation['spotify_preview_url']
                
                if ind == 0:
                    st.markdown(f"""
                    <div class='now-playing'>
                        <p style='color: #1ED760; font-size: 0.8em; margin: 0;'>▶ NOW PLAYING</p>
                        <p class='song-title'>{rec_song}</p>
                        <p class='artist-name'>{rec_artist}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if pd.notna(preview_url):
                        st.audio(preview_url)
                else:
                    st.markdown(f"""
                    <div class='song-card'>
                        <p><span class='rec-number'>#{ind}</span></p>
                        <p class='song-title'>{rec_song}</p>
                        <p class='artist-name'>{rec_artist}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if pd.notna(preview_url):
                        st.audio(preview_url)
        else:
            st.error(f"❌ Sorry, **{song_name.title()}** by **{artist_name.title()}** was not found in our database. Please try another song.")

    elif filtering_type == "Hybrid Recommender System":
        with st.spinner("Running hybrid recommendation engine..."):
            recommender = HybridRecommenderSystem(
                number_of_recommendations=k,
                weight_content_based=content_based_weight
            )
            
            recommendations = recommender.give_recommendations(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=filtered_data,
                transformed_matrix=transformed_hybrid_data,
                track_ids=track_ids,
                interaction_matrix=interaction_matrix
            )
        
        st.markdown(f"### Recommendations for **{song_name.title()}** by **{artist_name.title()}**")
        st.markdown(f"<span class='method-badge'>Hybrid (Content: {content_based_weight:.1f} + Collab: {diversity/10:.1f})</span>", unsafe_allow_html=True)
        st.markdown("")
        
        for ind, recommendation in recommendations.iterrows():
            rec_song = recommendation['name'].title()
            rec_artist = recommendation['artist'].title()
            preview_url = recommendation['spotify_preview_url']
            
            if ind == 0:
                st.markdown(f"""
                <div class='now-playing'>
                    <p style='color: #1ED760; font-size: 0.8em; margin: 0;'>▶ NOW PLAYING</p>
                    <p class='song-title'>{rec_song}</p>
                    <p class='artist-name'>{rec_artist}</p>
                </div>
                """, unsafe_allow_html=True)
                if pd.notna(preview_url):
                    st.audio(preview_url)
            else:
                st.markdown(f"""
                <div class='song-card'>
                    <p><span class='rec-number'>#{ind}</span></p>
                    <p class='song-title'>{rec_song}</p>
                    <p class='artist-name'>{rec_artist}</p>
                </div>
                """, unsafe_allow_html=True)
                if pd.notna(preview_url):
                    st.audio(preview_url)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6b7280; font-size: 0.85em;'>"
    "Powered by Hybrid Recommendation Engine • 30K+ Songs • 960K+ Users"
    "</p>",
    unsafe_allow_html=True
)
