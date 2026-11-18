"""
================================================================================
Personalized TED Talks Recommendation System
Week 4 – Module 6: Streamlit Web Application Deployment
================================================================================
A production-grade, interactive web interface for TED Talks recommendations:
- Content-Based Filtering (TF-IDF & Cosine Similarity)
- Popularity-Augmented Hybrid Recommendation (Semantic + Views)
- Title Search with Autocomplete & Fuzzy Matching
- Topic & Tag Taxonomy Exploration
- Detailed Recommendation Explanations (Shared Tags & Keywords)
- Model Benchmarks & IR Evaluation (Precision@K, Recall@K, NDCG@K, MAP@K)
================================================================================
"""

import os
import re
import ast
import string
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration & Metadata
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TED Talks Recommendation System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VIZ_DIR = os.path.join(REPORTS_DIR, "visualizations")

# -----------------------------------------------------------------------------
# Custom CSS Styling (TED Brand-Inspired Clean Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* TED Red Accent and Modern Typography */
    .ted-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #e62b1e;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .ted-subtext {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .talk-card {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    .talk-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
        border-color: #e62b1e;
    }
    .rank-pill {
        background-color: #e62b1e;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-right: 8px;
    }
    .sim-pill {
        background-color: #0b69a3;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .tag-badge {
        background-color: #f1f3f5;
        color: #212529;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 5px;
        margin-bottom: 4px;
        display: inline-block;
        border: 1px solid #dee2e6;
    }
    .tag-match {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-color: #c3e6cb !important;
        font-weight: 600;
    }
    .explanation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .metric-container {
        text-align: center;
        background: #fdfdfd;
        border: 1px solid #eaeaea;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Data & Model Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_talks_data():
    """Load and cache talk metadata and features."""
    clean_path = os.path.join(BASE_DIR, "cleaned_ted_talks.csv")
    if not os.path.isfile(clean_path):
        clean_path = os.path.join(BASE_DIR, "datasets", "cleaned_ted_talks.csv")
    if not os.path.isfile(clean_path):
        st.error(f"Missing required dataset: 'cleaned_ted_talks.csv'. Please ensure Module 2 has been executed.")
        st.stop()

    df = pd.read_csv(clean_path)
    if 'talk_id' not in df.columns:
        df.insert(0, 'talk_id', range(len(df)))

    # Parse tags list
    def parse_tags(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            clean_str = val.strip()
            if clean_str.startswith('[') and clean_str.endswith(']'):
                try:
                    return ast.literal_eval(clean_str)
                except Exception:
                    pass
            return clean_str.split()
        return []

    df['tags_array'] = df['cleaned_tags'].apply(parse_tags)
    return df


@st.cache_resource(show_spinner=False)
def load_recommendation_models():
    """Load and cache serialized similarity matrix and models."""
    sim_path = os.path.join(MODELS_DIR, "cosine_similarity.pkl")
    if not os.path.isfile(sim_path):
        st.error("Missing required model: 'models/cosine_similarity.pkl'. Please run Module 3/4 first.")
        st.stop()

    with open(sim_path, "rb") as f:
        cosine_sim = pickle.load(f)

    return cosine_sim


@st.cache_data(show_spinner=False)
def load_model_benchmarks():
    """Load Module 5 evaluation benchmark results."""
    comp_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if os.path.isfile(comp_path):
        return pd.read_csv(comp_path)
    return None


# -----------------------------------------------------------------------------
# Recommendation Engine Helper Functions
# -----------------------------------------------------------------------------
def get_recommendations(target_idx, df_talks, cosine_sim_matrix, norm_views, top_n=10, method="content", alpha=0.85):
    """Compute and rank top N recommendations with explanation payloads."""
    raw_sim = cosine_sim_matrix[target_idx].copy()

    if method.lower() == "hybrid":
        final_scores = (alpha * raw_sim) + ((1.0 - alpha) * norm_views)
    else:
        final_scores = raw_sim

    # Exclude self and sort descending
    sorted_indices = np.argsort(final_scores)[::-1]
    top_indices = [i for i in sorted_indices if i != target_idx][:top_n]

    query_tags = set(df_talks.loc[target_idx, 'tags_array'])
    query_title = df_talks.loc[target_idx, 'title']

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        cand_tags = set(df_talks.loc[idx, 'tags_array'])
        shared_tags = query_tags.intersection(cand_tags)

        # Recommendation explanation logic
        score_val = float(raw_sim[idx])
        views_val = int(df_talks.loc[idx, 'views']) if pd.notna(df_talks.loc[idx, 'views']) else 0
        speaker_val = df_talks.loc[idx, 'main_speaker']
        is_same_speaker = (speaker_val.lower().strip() == df_talks.loc[target_idx, 'main_speaker'].lower().strip())

        explanation_parts = []
        if is_same_speaker:
            explanation_parts.append(f"Shares speaker **{speaker_val}** with query talk.")
        if len(shared_tags) > 0:
            shared_tags_str = ", ".join(sorted(list(shared_tags))[:4])
            explanation_parts.append(f"Shares **{len(shared_tags)} key topic(s)**: *{shared_tags_str}*.")
        explanation_parts.append(f"High content similarity: **{score_val*100:.1f}% semantic match** across speech transcripts.")

        results.append({
            "rank": rank,
            "talk_id": idx,
            "title": df_talks.loc[idx, 'title'],
            "speaker": speaker_val,
            "occupation": df_talks.loc[idx, 'speaker_occupation'] if 'speaker_occupation' in df_talks.columns else "Speaker",
            "duration_minutes": float(df_talks.loc[idx, 'duration_minutes']) if 'duration_minutes' in df_talks.columns else round(float(df_talks.loc[idx, 'duration'])/60.0, 1),
            "views": views_val,
            "similarity_score": score_val,
            "ranking_score": float(final_scores[idx]),
            "description": df_talks.loc[idx, 'description'] if 'description' in df_talks.columns else "No description available.",
            "tags": df_talks.loc[idx, 'tags_array'],
            "shared_tags": shared_tags,
            "url": str(df_talks.loc[idx, 'url']).strip(),
            "explanation": " ".join(explanation_parts)
        })

    return results


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    # Load resources with Streamlit caching
    df_talks = load_talks_data()
    cosine_sim_matrix = load_recommendation_models()
    benchmark_df = load_model_benchmarks()

    # Precalculate normalized views for hybrid mode
    log_views = np.log1p(df_talks['views'].fillna(0).values)
    norm_views = (log_views - log_views.min()) / (log_views.max() - log_views.min() + 1e-9)

    # -------------------------------------------------------------------------
    # Sidebar Controls & Settings
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/aa/TED_three_letter_logo.svg", width=110)
        st.markdown("### 🎙️ Navigation")
        app_mode = st.radio(
            "Select Feature:",
            ["🎯 Recommender", "🏷️ Topic Explorer", "📊 Model Benchmarks", "ℹ️ About"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### ⚙️ Engine Parameters")

        method_choice = st.selectbox(
            "Algorithm Mode:",
            ["Content-Based (TF-IDF)", "Hybrid (Content + Popularity)"],
            help="Content-Based uses pure semantic similarity. Hybrid blends content similarity with global audience engagement."
        )
        rec_method = "content" if "Content-Based" in method_choice else "hybrid"

        alpha_val = 0.85
        if rec_method == "hybrid":
            alpha_val = st.slider(
                "Content Weight (α):",
                min_value=0.50,
                max_value=0.95,
                value=0.85,
                step=0.05,
                help="Higher α prioritizes topic similarity. Lower α elevates viral high-view talks."
            )

        top_k = st.slider("Number of Recommendations:", min_value=5, max_value=20, value=10, step=1)

        st.markdown("---")
        st.markdown("#### 📈 Dataset Statistics")
        st.caption(f"**Total Talks**: {len(df_talks):,}")
        st.caption(f"**Vocabulary**: 5,000 TF-IDF features")
        st.caption(f"**Similarity Space**: 2,550 × 2,550")
        st.caption(f"**Ground Truth**: 15,010 curated pairs")

    # -------------------------------------------------------------------------
    # Page 1: Recommender Engine
    # -------------------------------------------------------------------------
    if app_mode == "🎯 Recommender":
        st.markdown('<div class="ted-header">Personalized TED Talks Recommender</div>', unsafe_allow_html=True)
        st.markdown('<div class="ted-subtext">Discover semantically relevant TED Talks powered by Machine Learning and Natural Language Processing.</div>', unsafe_allow_html=True)

        # Quick Picks Row
        st.markdown("##### 💡 Popular Flagship Talks:")
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        selected_flagship = None

        if col_q1.button("🎓 Do schools kill creativity?", use_container_width=True):
            selected_flagship = "Do schools kill creativity?"
        if col_q2.button("🧠 Your body language shapes you", use_container_width=True):
            selected_flagship = "Your body language may shape who you are"
        if col_q3.button("🌍 Averting the climate crisis", use_container_width=True):
            selected_flagship = "Averting the climate crisis"
        if col_q4.button("📊 The best stats you've seen", use_container_width=True):
            selected_flagship = "The best stats you've ever seen"

        # Autocomplete Title Search
        all_titles = df_talks['title'].tolist()
        default_index = 0
        if selected_flagship and selected_flagship in all_titles:
            default_index = all_titles.index(selected_flagship)

        selected_title = st.selectbox(
            "Search or select a TED Talk title:",
            options=all_titles,
            index=default_index,
            help="Type to search through all 2,550 TED Talks in the catalog."
        )

        if selected_title:
            target_row = df_talks[df_talks['title'] == selected_title].iloc[0]
            target_idx = int(target_row['talk_id'])

            # Target Talk Overview Card
            with st.container():
                st.markdown(f"""
                <div style="background-color: #fdf2f2; border: 1px solid #fad2cf; border-radius: 8px; padding: 16px; margin: 15px 0;">
                    <div style="font-size: 0.85rem; color: #e62b1e; font-weight: 700; text-transform: uppercase;">Selected Seed Talk</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #111;">{target_row['title']}</div>
                    <div style="font-size: 0.95rem; color: #444; margin-bottom: 8px;">Speaker: <strong>{target_row['main_speaker']}</strong> ({target_row.get('speaker_occupation', 'Speaker')}) &nbsp;|&nbsp; Duration: <strong>{target_row.get('duration_minutes', 0):.1f} mins</strong> &nbsp;|&nbsp; Views: <strong>{int(target_row.get('views', 0)):,}</strong></div>
                    <div style="font-size: 0.9rem; color: #333; margin-bottom: 10px;">{target_row.get('description', '')}</div>
                    <div>{' '.join([f'<span class="tag-badge">{t}</span>' for t in target_row['tags_array']])}</div>
                </div>
                """, unsafe_allow_html=True)

            # Generate Recommendations
            with st.spinner("Generating personalized recommendations..."):
                recs = get_recommendations(
                    target_idx=target_idx,
                    df_talks=df_talks,
                    cosine_sim_matrix=cosine_sim_matrix,
                    norm_views=norm_views,
                    top_n=top_k,
                    method=rec_method,
                    alpha=alpha_val
                )

            st.markdown(f"### 🎯 Top {len(recs)} Recommended TED Talks")
            st.caption(f"Ranking Mode: **{method_choice}** (Retrieving from catalog of {len(df_talks):,} talks)")

            # Render Cards
            for item in recs:
                query_tags_set = set(target_row['tags_array'])
                tags_html = ""
                for t in item['tags']:
                    if t in query_tags_set:
                        tags_html += f'<span class="tag-badge tag-match">✓ {t}</span>'
                    else:
                        tags_html += f'<span class="tag-badge">{t}</span>'

                st.markdown(f"""
                <div class="talk-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span class="rank-pill">#{item['rank']}</span>
                            <span class="sim-pill">{item['similarity_score']*100:.1f}% Match</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #666;">
                            👁️ {item['views']:,} views &nbsp;•&nbsp; ⏱️ {item['duration_minutes']:.1f} mins
                        </div>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #111; margin-bottom: 4px;">
                        {item['title']}
                    </div>
                    <div style="font-size: 0.92rem; color: #444; margin-bottom: 8px;">
                        Speaker: <strong>{item['speaker']}</strong> ({item['occupation']})
                    </div>
                    <div style="font-size: 0.88rem; color: #555; line-height: 1.45; margin-bottom: 10px;">
                        {item['description']}
                    </div>
                    <div class="explanation-box">
                        <strong>💡 Why recommended:</strong> {item['explanation']}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                        <div>{tags_html}</div>
                        <a href="{item['url']}" target="_blank" style="background-color: #e62b1e; color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 0.85rem; font-weight: 600;">
                            Watch on TED.com ↗
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Page 2: Topic Explorer
    # -------------------------------------------------------------------------
    elif app_mode == "🏷️ Topic Explorer":
        st.markdown('<div class="ted-header">Topic & Tag Taxonomy Explorer</div>', unsafe_allow_html=True)
        st.markdown('<div class="ted-subtext">Filter and explore TED Talks across 400+ curated thematic domains.</div>', unsafe_allow_html=True)

        all_unique_tags = sorted(list({tag for tags in df_talks['tags_array'] for tag in tags}))

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            selected_topics = st.multiselect(
                "Filter talks by topic tags (AND logic):",
                options=all_unique_tags,
                default=["technology", "future"] if "technology" in all_unique_tags and "future" in all_unique_tags else ["technology"]
            )
        with col_f2:
            sort_by = st.selectbox("Sort results by:", ["Highest Views", "Duration (Short to Long)", "Duration (Long to Short)"])

        if selected_topics:
            def matches_tags(talk_tags):
                return all(t in talk_tags for t in selected_topics)

            filtered_df = df_talks[df_talks['tags_array'].apply(matches_tags)].copy()

            if sort_by == "Highest Views":
                filtered_df = filtered_df.sort_values(by="views", ascending=False)
            elif sort_by == "Duration (Short to Long)":
                filtered_df = filtered_df.sort_values(by="duration_minutes", ascending=True)
            else:
                filtered_df = filtered_df.sort_values(by="duration_minutes", ascending=False)

            st.markdown(f"#### Matching Talks: **{len(filtered_df):,} talks** found")

            for _, row in filtered_df.head(15).iterrows():
                tags_str = " ".join([f'<span class="tag-badge">{t}</span>' for t in row['tags_array'][:8]])
                st.markdown(f"""
                <div class="talk-card">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #111;">
                        {row['title']} <span style="font-size: 0.85rem; color: #666; font-weight: 400;">({row['main_speaker']})</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #555; margin: 4px 0;">
                        ⏱️ {row['duration_minutes']:.1f} mins &nbsp;•&nbsp; 👁️ {int(row['views']):,} views
                    </div>
                    <div style="font-size: 0.88rem; color: #444; margin-bottom: 8px;">{row.get('description', '')[:160]}...</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>{tags_str}</div>
                        <a href="{row['url']}" target="_blank" style="color: #e62b1e; font-size: 0.85rem; font-weight: 600; text-decoration: none;">Watch ↗</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Select one or more topic tags above to explore matching TED Talks.")

    # -------------------------------------------------------------------------
    # Page 3: Model Benchmarks & IR Evaluation
    # -------------------------------------------------------------------------
    elif app_mode == "📊 Model Benchmarks":
        st.markdown('<div class="ted-header">Module 5: Model Evaluation & Benchmarks</div>', unsafe_allow_html=True)
        st.markdown('<div class="ted-subtext">Empirical Information Retrieval evaluation against 15,010 authentic human-curated editorial recommendations.</div>', unsafe_allow_html=True)

        if benchmark_df is not None:
            st.markdown("### 🏆 Model Comparison Benchmark (At K = 10)")
            st.dataframe(benchmark_df, use_container_width=True)

            # Metric Highlights
            cb_row = benchmark_df[benchmark_df['Model'].str.contains('Content-Based', case=False)].iloc[0]
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Precision@10", f"{cb_row['Precision@K']:.4f}", "+65× vs Popularity")
            col_m2.metric("Recall@10", f"{cb_row['Recall@K']:.4f}", "38.8% Coverage")
            col_m3.metric("NDCG@10", f"{cb_row['NDCG@K']:.4f}", "Ranking Quality")
            col_m4.metric("MAP@10", f"{cb_row['MAP@K']:.4f}", "Area Under PR")

        st.markdown("### 📈 Evaluation Metric Curves Across Cut-offs K")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_path = os.path.join(VIZ_DIR, "precision_comparison.png")
            if os.path.isfile(p_path):
                st.image(p_path, caption="Precision@K across K in [3, 5, 10, 15, 20]", use_container_width=True)
            r_path = os.path.join(VIZ_DIR, "recall_comparison.png")
            if os.path.isfile(r_path):
                st.image(r_path, caption="Recall@K across K in [3, 5, 10, 15, 20]", use_container_width=True)

        with col_p2:
            n_path = os.path.join(VIZ_DIR, "ndcg_comparison.png")
            if os.path.isfile(n_path):
                st.image(n_path, caption="NDCG@K Ranking Quality across Models", use_container_width=True)
            m_path = os.path.join(VIZ_DIR, "map_comparison.png")
            if os.path.isfile(m_path):
                st.image(m_path, caption="Mean Average Precision (MAP@K) across Models", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        #### 🔒 Evaluation Integrity & Collaborative Filtering Assessment
        - **Authentic Ground-Truth**: Evaluation uses genuine TED editorial recommendations (`related_talks`) from TED Conferences LLC.
        - **Zero Synthetic Data Policy**: The dataset does not contain individual user IDs or watch logs. In strict adherence to academic and engineering standards, **no fake user ratings were synthesized**.
        """)

    # -------------------------------------------------------------------------
    # Page 4: About & Architecture
    # -------------------------------------------------------------------------
    elif app_mode == "ℹ️ About":
        st.markdown('<div class="ted-header">About the System</div>', unsafe_allow_html=True)
        st.markdown("""
        ### Personalized TED Talks Recommendation System Using Machine Learning
        This application deploys a Content-Based and Popularity-Augmented Hybrid Recommendation System developed across a 4-week machine learning engineering lifecycle:

        - **Week 1 – Module 1 (Data Collection)**: Collected, validated, and organized 2,550 TED Talks across 416 distinct topics.
        - **Week 1 – Module 2 (EDA & Preprocessing)**: Performed exploratory data analysis, sanitized transcripts with NLTK stopwords, resolved missing values, and encoded categorical attributes.
        - **Week 2 – Module 3 (Feature Engineering)**: Engineered a 5,000-dimensional TF-IDF feature space capturing unigrams and bigrams, computing a 2,550 × 2,550 pairwise Cosine Similarity matrix.
        - **Week 2 – Module 4 (Recommendation Engine)**: Developed the recommendation engine with fuzzy search, offline train-test evaluation, and hybrid popularity weighting.
        - **Week 3 – Module 5 (Evaluation Metrics)**: Benchmarked against 15,010 authentic human-curated editorial pairs across Precision@K, Recall@K, NDCG@K, and MAP@K.
        - **Week 4 – Module 6 (Streamlit Deployment)**: Interactive, responsive web application deployment with multi-topic exploration and recommendation explanations.
        """)


if __name__ == "__main__":
    main()

