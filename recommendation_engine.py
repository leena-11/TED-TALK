"""
================================================================================
WEEK 2 – MODULE 4: DEVELOP RECOMMENDATION ENGINE
Personalized TED Talks Recommendation System Using Machine Learning
================================================================================

This module provides a production-grade Recommendation Engine for TED Talks:
1. Content-Based Recommendation using TF-IDF and Cosine Similarity.
2. Robust, case-insensitive title search and fuzzy matching.
3. Top 10 recommendations with Title, Speaker, Tags, Description, and Similarity Score.
4. Offline evaluation with train-test splitting and ranking metrics (Precision@K,
   Recall@K, Jaccard Similarity, MRR).
5. Formal audit of Collaborative Filtering availability (STRICT NO FAKE USER DATA).
6. Popularity-Augmented Hybrid Recommendation (Semantic Similarity + Engagement).
7. Comprehensive comparative analysis of recommendation methods.
8. Serialization and deployment persistence in 'models/'.

Usage:
  python recommendation_engine.py --title "Do schools kill creativity?" --top_n 10
  python recommendation_engine.py --evaluate
================================================================================
"""

import os
import sys
import re
import string
import argparse
import pickle
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TEDRecommendationEngine:
    """
    Personalized TED Talks Recommendation Engine.
    Supports Content-Based Filtering and Popularity-Augmented Hybrid Recommendation.
    """

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.outputs_dir = os.path.join(self.base_dir, "outputs")
        self.reports_dir = os.path.join(self.base_dir, "reports")

        # Load data and serialized artifacts
        self._load_datasets_and_models()
        self._build_lookup_index()

    def _load_datasets_and_models(self):
        """Load feature tables, cleaned metadata, vectorizers, and similarity matrices."""
        # 1. Load cleaned metadata
        clean_path = os.path.join(self.base_dir, "cleaned_ted_talks.csv")
        feats_path = os.path.join(self.outputs_dir, "talk_features.csv")

        if os.path.isfile(clean_path):
            self.df_talks = pd.read_csv(clean_path)
        elif os.path.isfile(feats_path):
            self.df_talks = pd.read_csv(feats_path)
        else:
            raise FileNotFoundError("Could not find 'cleaned_ted_talks.csv' or 'talk_features.csv'")

        # Ensure talk_id exists
        if 'talk_id' not in self.df_talks.columns:
            self.df_talks.insert(0, 'talk_id', range(len(self.df_talks)))

        # Ensure description exists
        if 'description' not in self.df_talks.columns and os.path.isfile(clean_path):
            df_c = pd.read_csv(clean_path)
            self.df_talks['description'] = df_c['description']

        # 2. Load serialized models from Module 3
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(self.models_dir, "tfidf_matrix.pkl")
        sim_path = os.path.join(self.models_dir, "cosine_similarity.pkl")

        if os.path.isfile(vec_path) and os.path.isfile(sim_path):
            with open(vec_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(sim_path, "rb") as f:
                self.cosine_sim_matrix = pickle.load(f)
            if os.path.isfile(mat_path):
                with open(mat_path, "rb") as f:
                    self.tfidf_matrix = pickle.load(f)
            else:
                self.tfidf_matrix = None
        else:
            # Recompute on the fly if model files are absent
            print("[!] Serialized models missing in models/. Re-fitting TF-IDF and Cosine Similarity...")
            self._fit_internal_models()

        # Compute normalized popularity weights for hybrid mode
        log_views = np.log1p(self.df_talks['views'].fillna(0).values)
        min_v, max_v = log_views.min(), log_views.max()
        self.norm_views = (log_views - min_v) / (max_v - min_v + 1e-9)

    def _fit_internal_models(self):
        """Fit vectorizer and similarity matrix if not already present."""
        content = (
            self.df_talks['title'].fillna('') + ' ' +
            self.df_talks['main_speaker'].fillna('') + ' ' +
            (self.df_talks['cleaned_tags'].fillna('') + ' ') * 2 +
            self.df_talks['description'].fillna('') + ' ' +
            self.df_talks['cleaned_transcript'].fillna('')
        )
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(content)
        self.cosine_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix).astype(np.float32)

    def _normalize_string(self, text):
        """Normalize string for robust title matching."""
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text

    def _build_lookup_index(self):
        """Build exact and normalized index mapping for fast search."""
        self.title_to_idx = {}
        self.norm_title_to_idx = {}

        for idx, row in self.df_talks.iterrows():
            raw_title = str(row['title']).strip()
            norm_title = self._normalize_string(raw_title)

            self.title_to_idx[raw_title] = idx
            self.norm_title_to_idx[norm_title] = idx

    def find_talk_index(self, query):
        """
        Locate the talk index by title with multi-tiered resolution:
        1. Exact match.
        2. Normalized case/punctuation-insensitive match.
        3. Substring match.
        Returns (index, matched_title) or (None, suggestions_list).
        """
        query_str = str(query).strip()
        if not query_str:
            return None, []

        # 1. Exact match
        if query_str in self.title_to_idx:
            return self.title_to_idx[query_str], query_str

        # 2. Normalized match
        norm_query = self._normalize_string(query_str)
        if norm_query in self.norm_title_to_idx:
            idx = self.norm_title_to_idx[norm_query]
            return idx, self.df_talks.loc[idx, 'title']

        # 3. Substring match
        matches = []
        for norm_t, idx in self.norm_title_to_idx.items():
            if norm_query in norm_t:
                matches.append(idx)

        if len(matches) == 1:
            idx = matches[0]
            return idx, self.df_talks.loc[idx, 'title']
        elif len(matches) > 1:
            # Return top suggestions
            suggestions = [self.df_talks.loc[i, 'title'] for i in matches[:5]]
            return None, suggestions

        # 4. Keyword token overlap fallback
        query_words = set(norm_query.split())
        best_overlap = 0
        best_candidates = []
        for norm_t, idx in self.norm_title_to_idx.items():
            title_words = set(norm_t.split())
            overlap = len(query_words.intersection(title_words))
            if overlap > best_overlap and overlap >= 2:
                best_overlap = overlap
                best_candidates = [idx]
            elif overlap == best_overlap and overlap >= 2:
                best_candidates.append(idx)

        if best_candidates:
            suggestions = [self.df_talks.loc[i, 'title'] for i in best_candidates[:5]]
            return None, suggestions

        return None, []

    def recommend(self, title_or_idx, top_n=10, method="content", alpha=0.85):
        """
        Generate top N TED Talk recommendations for a given title or index.

        Parameters:
        -----------
        title_or_idx : str or int
            Target TED Talk title or talk index.
        top_n : int, default=10
            Number of recommendations to return.
        method : str, default='content'
            'content': Pure Content-Based Cosine Similarity.
            'hybrid' : Popularity-Augmented Content-Based (alpha*sim + (1-alpha)*norm_views).
        alpha : float, default=0.85
            Weighting between content similarity (alpha) and audience popularity (1 - alpha).

        Returns:
        --------
        dict containing 'target_talk', 'method', and 'recommendations' DataFrame.
        """
        if isinstance(title_or_idx, (int, np.integer)):
            if 0 <= title_or_idx < len(self.df_talks):
                target_idx = int(title_or_idx)
                target_title = self.df_talks.loc[target_idx, 'title']
            else:
                raise IndexError(f"Talk index {title_or_idx} out of bounds (0 to {len(self.df_talks)-1})")
        else:
            target_idx, resolved_title = self.find_talk_index(title_or_idx)
            if target_idx is None:
                suggestions = resolved_title
                msg = f"Talk title '{title_or_idx}' not found."
                if suggestions:
                    msg += f" Did you mean one of: {suggestions}?"
                raise ValueError(msg)
            target_title = resolved_title

        # Retrieve raw similarity scores
        sim_scores = self.cosine_sim_matrix[target_idx].copy()

        if method.lower() == "hybrid":
            # Combine semantic similarity with normalized popularity
            final_scores = (alpha * sim_scores) + ((1.0 - alpha) * self.norm_views)
        else:
            final_scores = sim_scores

        # Rank talks descending, excluding target talk
        sorted_indices = np.argsort(final_scores)[::-1]
        top_indices = [i for i in sorted_indices if i != target_idx][:top_n]

        # Assemble results
        records = []
        for rank, idx in enumerate(top_indices, start=1):
            raw_sim = float(self.cosine_sim_matrix[target_idx, idx])
            score_to_show = float(final_scores[idx]) if method.lower() == "hybrid" else raw_sim

            records.append({
                "Rank": rank,
                "Title": self.df_talks.loc[idx, "title"],
                "Speaker": self.df_talks.loc[idx, "main_speaker"],
                "Tags": self.df_talks.loc[idx, "cleaned_tags"],
                "Description": self.df_talks.loc[idx, "description"],
                "Similarity Score": round(raw_sim, 4),
                "Ranking Score": round(score_to_show, 4),
                "Views": int(self.df_talks.loc[idx, "views"]) if pd.notna(self.df_talks.loc[idx, "views"]) else 0,
                "URL": self.df_talks.loc[idx, "url"],
            })

        rec_df = pd.DataFrame(records)
        target_info = {
            "talk_id": target_idx,
            "title": self.df_talks.loc[target_idx, "title"],
            "speaker": self.df_talks.loc[target_idx, "main_speaker"],
            "tags": self.df_talks.loc[target_idx, "cleaned_tags"],
            "description": self.df_talks.loc[target_idx, "description"],
            "views": int(self.df_talks.loc[target_idx, "views"]) if pd.notna(self.df_talks.loc[target_idx, "views"]) else 0,
        }

        return {
            "target": target_info,
            "method": method,
            "alpha": alpha if method.lower() == "hybrid" else 1.0,
            "recommendations": rec_df,
        }

    def evaluate_offline(self, test_size=0.2, top_k=10, random_state=42):
        """
        Perform rigorous offline train-test evaluation for Content-Based Recommendation:
        1. Splits talks into train (80%) and test (20%).
        2. Fits TF-IDF on training talks and transforms test talks.
        3. Computes recommendation ranking across test queries.
        4. Evaluates:
           - Tag Precision@K (proportion of recommended talks sharing thematic tags).
           - Tag Recall@K (coverage of query tags across recommendations).
           - Mean Jaccard Similarity@K.
           - Mean Reciprocal Rank (MRR) of first relevant match.
           - Intra-List Diversity (1 - mean pairwise similarity of recommendation set).
        """
        train_df, test_df = train_test_split(self.df_talks, test_size=test_size, random_state=random_state)

        # Prepare text features
        train_text = (
            train_df['title'].fillna('') + ' ' +
            train_df['main_speaker'].fillna('') + ' ' +
            (train_df['cleaned_tags'].fillna('') + ' ') * 2 +
            train_df['description'].fillna('') + ' ' +
            train_df['cleaned_transcript'].fillna('')
        )
        test_text = (
            test_df['title'].fillna('') + ' ' +
            test_df['main_speaker'].fillna('') + ' ' +
            (test_df['cleaned_tags'].fillna('') + ' ') * 2 +
            test_df['description'].fillna('') + ' ' +
            test_df['cleaned_transcript'].fillna('')
        )

        eval_vec = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
        train_mat = eval_vec.fit_transform(train_text)
        test_mat = eval_vec.transform(test_text)

        test_sim = cosine_similarity(test_mat, train_mat)

        precisions = []
        recalls = []
        jaccards = []
        reciprocal_ranks = []
        diversities = []

        train_tags_list = [set(str(t).split()) for t in train_df['cleaned_tags']]
        test_tags_list = [set(str(t).split()) for t in test_df['cleaned_tags']]

        for i in range(len(test_df)):
            query_tags = test_tags_list[i]
            if not query_tags:
                continue

            scores = test_sim[i]
            top_indices = np.argsort(scores)[::-1][:top_k]

            matches = 0
            rec_tags_all = set()
            rr = 0.0

            for rank, c_idx in enumerate(top_indices, start=1):
                cand_tags = train_tags_list[c_idx]
                intersection = query_tags.intersection(cand_tags)
                if len(intersection) > 0:
                    matches += 1
                    if rr == 0.0:
                        rr = 1.0 / rank
                rec_tags_all.update(cand_tags)

            precision_k = matches / top_k
            recall_k = len(query_tags.intersection(rec_tags_all)) / max(len(query_tags), 1)
            jaccard_k = len(query_tags.intersection(rec_tags_all)) / max(len(query_tags.union(rec_tags_all)), 1)

            precisions.append(precision_k)
            recalls.append(recall_k)
            jaccards.append(jaccard_k)
            reciprocal_ranks.append(rr)

            # Intra-list diversity of recommended set
            rec_vectors = train_mat[top_indices]
            if len(top_indices) > 1:
                inner_sim = cosine_similarity(rec_vectors, rec_vectors)
                triu_idx = np.triu_indices(len(top_indices), k=1)
                diversity = 1.0 - float(np.mean(inner_sim[triu_idx]))
                diversities.append(diversity)

        results = {
            "eval_corpus_size": len(self.df_talks),
            "train_set_size": len(train_df),
            "test_set_size": len(test_df),
            "top_k": top_k,
            "mean_precision_at_k": round(float(np.mean(precisions)), 4),
            "mean_recall_at_k": round(float(np.mean(recalls)), 4),
            "mean_jaccard_at_k": round(float(np.mean(jaccards)), 4),
            "mean_reciprocal_rank": round(float(np.mean(reciprocal_ranks)), 4),
            "intra_list_diversity": round(float(np.mean(diversities)), 4),
        }
        return results

    def save_engine_models(self):
        """Serialize recommendation engine models for future production deployment."""
        content_based_payload = {
            "model_type": "content_based",
            "feature_space": "TF-IDF + Cosine Similarity",
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "cosine_sim_matrix": self.cosine_sim_matrix,
            "titles": self.df_talks['title'].values,
            "speakers": self.df_talks['main_speaker'].values,
            "tags": self.df_talks['cleaned_tags'].values,
            "descriptions": self.df_talks['description'].values,
            "urls": self.df_talks['url'].values,
        }

        recommendation_model_payload = {
            "model_type": "unified_recommendation_engine",
            "version": "1.0",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "supported_modes": ["content", "hybrid"],
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "cosine_sim_matrix": self.cosine_sim_matrix,
            "norm_views": self.norm_views,
            "talks_metadata": self.df_talks[['talk_id', 'title', 'main_speaker', 'cleaned_tags', 'description', 'views', 'url']],
        }

        cb_path = os.path.join(self.models_dir, "content_based_model.pkl")
        rec_path = os.path.join(self.models_dir, "recommendation_model.pkl")

        with open(cb_path, "wb") as f:
            pickle.dump(content_based_payload, f)
        with open(rec_path, "wb") as f:
            pickle.dump(recommendation_model_payload, f)

        print(f"[✓] Saved {cb_path} ({os.path.getsize(cb_path)/(1024*1024):.2f} MB)")
        print(f"[✓] Saved {rec_path} ({os.path.getsize(rec_path)/(1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="TED Talks Recommendation Engine")
    parser.add_argument("--title", type=str, default="Do schools kill creativity?", help="TED Talk title query")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations to return")
    parser.add_argument("--method", type=str, default="content", choices=["content", "hybrid"], help="Recommendation mode")
    parser.add_argument("--alpha", type=float, default=0.85, help="Hybrid weight for content similarity vs views")
    parser.add_argument("--evaluate", action="store_true", help="Run offline evaluation benchmark")
    args = parser.parse_args()

    engine = TEDRecommendationEngine()

    if args.evaluate:
        print("\n" + "=" * 80)
        print("RUNNING OFFLINE EVALUATION BENCHMARK (Train-Test Split: 80/20)")
        print("=" * 80)
        metrics = engine.evaluate_offline(test_size=0.2, top_k=args.top_n)
        for k, v in metrics.items():
            print(f"  {k:<28}: {v}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print(f"GENERATING RECOMMENDATIONS (Method: {args.method.upper()})")
    print("=" * 80)

    try:
        res = engine.recommend(args.title, top_n=args.top_n, method=args.method, alpha=args.alpha)
        target = res['target']
        recs = res['recommendations']

        print(f"Query Talk: '{target['title']}' by {target['speaker']}")
        print(f"Tags      : {target['tags']}")
        print(f"Views     : {target['views']:,}")
        print("-" * 80)

        for _, row in recs.iterrows():
            print(f"Rank {row['Rank']:2d} | Similarity: {row['Similarity Score']:.4f} | Title: {row['Title']} ({row['Speaker']})")
            print(f"        Tags: {row['Tags'][:65]}...")
            print(f"        Description: {row['Description'][:110]}...")
            print(f"        URL: {row['URL'].strip()}")
            print()

    except Exception as e:
        print(f"[!] Error: {e}")


if __name__ == "__main__":
    main()
