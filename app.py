"""
app.py — Main Streamlit application for Voice Cloning Detection.

Pages:
  🏠 Home Dashboard
  📊 Dataset Analysis
  🎵 Audio Analysis
  🤖 Model Training
  🛡️  Voice Cloning Detection
  📈 Evaluation Results
"""

from __future__ import annotations

import sys
import json
import time
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure project root on sys.path ──────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    MANIFEST_PATH, MODEL_PATH, SCALER_PATH, METRICS_PATH,
    CONFUSION_MATRIX_PATH, FEATURE_IMP_PATH, ROC_CURVE_PATH,
    FEATURE_CFG, WAVEFAKE_ROOT, SPOOF_THRESHOLD,
    LABEL_MAP, N_FEATURES, FEATURE_NAMES,
    SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, DEMO_MODE, SMALL_DATASET,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Voice Cloning Detection | AI Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — premium dark UI
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --bg-glass:      rgba(28,33,40,0.85);
    --accent-blue:   #2f81f7;
    --accent-green:  #3fb950;
    --accent-red:    #f85149;
    --accent-orange: #e3b341;
    --accent-purple: #a371f7;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
    --border:        rgba(48,54,61,0.8);
    --shadow:        0 8px 32px rgba(0,0,0,0.4);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid var(--border);
}

/* Cards */
.vc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: transform 0.2s, box-shadow 0.2s;
}
.vc-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.5); }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #1a1f2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(47,129,247,0.15) 0%, transparent 70%);
}
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #2f81f7, #a371f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p  { color: var(--text-muted); font-size: 1.05rem; margin-top: 0.5rem; }

/* Stat tiles */
.stat-tile {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-tile .val { font-size: 2rem; font-weight: 700; color: var(--accent-blue); }
.stat-tile .lbl { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem; }

/* Status badges */
.badge { display:inline-block; padding:0.25rem 0.75rem; border-radius:20px;
         font-size:0.78rem; font-weight:600; }
.badge-green  { background:#1b4332; color:#3fb950; border:1px solid #2ea043; }
.badge-red    { background:#3d1a1a; color:#f85149; border:1px solid #da3633; }
.badge-orange { background:#3d2b1a; color:#e3b341; border:1px solid #c9960d; }
.badge-blue   { background:#102a4c; color:#2f81f7; border:1px solid #1f6feb; }

/* Alert boxes */
.alert-danger {
    background: rgba(248,81,73,0.12);
    border: 1px solid rgba(248,81,73,0.4);
    border-left: 4px solid #f85149;
    border-radius: 8px; padding: 1rem 1.25rem;
    color: #ffa198;
}
.alert-success {
    background: rgba(63,185,80,0.12);
    border: 1px solid rgba(63,185,80,0.4);
    border-left: 4px solid #3fb950;
    border-radius: 8px; padding: 1rem 1.25rem;
    color: #7ee787;
}
.alert-info {
    background: rgba(47,129,247,0.12);
    border: 1px solid rgba(47,129,247,0.4);
    border-left: 4px solid #2f81f7;
    border-radius: 8px; padding: 1rem 1.25rem;
    color: #79c0ff;
}

/* Code / mono */
.mono { font-family:'Fira Code', monospace; font-size:0.85rem; color:#79c0ff; }

/* Probability bar */
.prob-bar-bg {
    background: var(--bg-secondary); border-radius:8px; height:18px;
    width:100%; border:1px solid var(--border); overflow:hidden;
}
.prob-bar-fill { height:100%; border-radius:8px; transition:width 0.5s ease; }

/* Section headings */
.section-title {
    font-size:1.3rem; font-weight:700; color:var(--text-primary);
    border-bottom:2px solid var(--accent-blue);
    padding-bottom:0.4rem; margin-bottom:1.2rem;
}

/* Streamlit overrides */
.stButton>button {
    background: linear-gradient(90deg, #2f81f7, #a371f7);
    color: white; border:none; border-radius:8px;
    padding:0.5rem 1.5rem; font-weight:600; font-size:0.95rem;
    transition: all 0.2s;
}
.stButton>button:hover { opacity:0.88; transform:translateY(-1px); }
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border:1px solid var(--border);
    border-radius:10px; padding:0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:2.5rem;">🛡️</div>
        <div style="font-weight:800; font-size:1.1rem; color:#e6edf3;">VoiceGuard AI</div>
        <div style="font-size:0.75rem; color:#8b949e; margin-top:0.2rem;">Voice Cloning Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "📊 Dataset Analysis",
            "🎵 Audio Analysis",
            "🤖 Model Training",
            "🛡️ Voice Cloning Detection",
            "📈 Evaluation Results",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#8b949e; text-align:center; line-height:1.6;">
        <b>ML Technique:</b> Supervised Classification<br>
        <b>Model:</b> Random Forest Classifier<br>
        <b>Dataset:</b> WaveFake<br>
        <b>Features:</b> MFCC + Spectral (31-dim)<br>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_manifest_cached():
    if not MANIFEST_PATH.exists():
        return None
    return pd.read_csv(MANIFEST_PATH)

@st.cache_data(ttl=60)
def _load_metrics_cached():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)

@st.cache_data(ttl=60)
def _load_feature_cfg_cached():
    if not FEATURE_CFG.exists():
        return None
    with open(FEATURE_CFG) as f:
        return json.load(f)


def _status_badge(ok: bool, ok_text="Ready", fail_text="Not Ready") -> str:
    cls  = "badge-green" if ok else "badge-red"
    text = ok_text if ok else fail_text
    return f'<span class="badge {cls}">{text}</span>'

def _render_prob_bar(prob: float, color: str) -> str:
    pct = round(prob * 100, 1)
    return f"""
    <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <small style="color:#8b949e;">{pct}%</small>
    """


# ═════════════════════════════════════════════
# PAGE 1 — HOME DASHBOARD
# ═════════════════════════════════════════════

if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>🛡️ AI-Powered Voice Cloning Detection</h1>
        <p>Real-Time Detection & Prevention of Voice Cloning Impersonation Attacks<br>
        Using Supervised Machine Learning (Random Forest Classifier)</p>
    </div>
    """, unsafe_allow_html=True)

    # Status strip
    dataset_ok  = MANIFEST_PATH.exists()
    model_ok    = MODEL_PATH.exists() and SCALER_PATH.exists()
    metrics_ok  = METRICS_PATH.exists()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="vc-card">
            <b>📂 Dataset</b><br>
            {_status_badge(dataset_ok, "Manifest Ready", "Not Loaded")}
            <p style="margin:0.5rem 0 0; color:#8b949e; font-size:0.85rem;">
            WaveFake — genuine LJSpeech + synthetic vocoders</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="vc-card">
            <b>🤖 Model</b><br>
            {_status_badge(model_ok, "Trained & Saved", "Not Trained")}
            <p style="margin:0.5rem 0 0; color:#8b949e; font-size:0.85rem;">
            Random Forest Classifier · 200 estimators</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="vc-card">
            <b>📈 Evaluation</b><br>
            {_status_badge(metrics_ok, "Results Available", "Awaiting Training")}
            <p style="margin:0.5rem 0 0; color:#8b949e; font-size:0.85rem;">
            Acc · Prec · Rec · F1 · AUC on held-out test set</p>
        </div>""", unsafe_allow_html=True)

    # Metrics strip (if available)
    metrics = _load_metrics_cached()
    if metrics:
        st.markdown('<div class="section-title">📊 Latest Measured Results</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        for col, key, label, fmt in [
            (m1, "accuracy",  "Accuracy",  ".4f"),
            (m2, "precision", "Precision", ".4f"),
            (m3, "recall",    "Recall",    ".4f"),
            (m4, "f1_score",  "F1-Score",  ".4f"),
            (m5, "roc_auc",   "ROC-AUC",   ".4f"),
        ]:
            val = metrics.get(key)
            col.metric(label, f"{val:{fmt}}" if val is not None else "N/A")

    # Objective & methodology
    st.markdown('<div class="section-title">🎯 Project Objective</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="vc-card">
        <b>Problem Statement</b><br><br>
        Voice cloning technology can synthesise a convincing replica of any person's
        voice from just a few seconds of reference audio. Malicious actors exploit this
        to impersonate individuals in phone fraud, social engineering, and
        authentication bypass attacks.<br><br>
        <b>Proposed Solution</b><br><br>
        A <span class="mono">Random Forest Classifier</span> trained on acoustic
        features (MFCC + spectral descriptors) extracted from genuine and AI-generated
        speech samples, enabling real-time detection of suspected spoof recordings.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="vc-card">
        <b>Methodology Pipeline</b><br><br>
        <ol style="color:#8b949e; line-height:2;">
          <li>Load WaveFake dataset (genuine + synthetic splits)</li>
          <li>Extract 31-dimensional acoustic feature vector per file</li>
          <li>Fit <code>StandardScaler</code> on training data only</li>
          <li>Train <code>RandomForestClassifier</code> (200 trees)</li>
          <li>Evaluate on stratified held-out test set (20%)</li>
          <li>Serve real-time predictions via Streamlit UI</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    # Cybersecurity note
    st.markdown("""
    <div class="alert-info">
    <b>⚠️ Cybersecurity Disclaimer</b><br>
    This system is a defensive research tool. The model output is a probability
    estimate and should <b>never</b> be the sole basis for high-stakes decisions
    (financial transactions, physical access, legal proceedings). Always perform
    independent identity verification when the stakes are high.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 2 — DATASET ANALYSIS
# ═════════════════════════════════════════════

elif page == "📊 Dataset Analysis":
    st.markdown('<div class="hero"><h1>📊 Dataset Analysis</h1><p>WaveFake dataset exploration and statistics</p></div>', unsafe_allow_html=True)

    manifest = _load_manifest_cached()

    if manifest is None:
        # ── Check if any real audio exists ─────
        real_data_exists = (
            (WAVEFAKE_ROOT / "LJSpeech-1.1" / "wavs").exists()
            or (WAVEFAKE_ROOT / "generated_audio").exists()
        )

        if not real_data_exists:
            st.markdown("""
            <div style="background:rgba(47,129,247,0.1);border:1px solid rgba(47,129,247,0.4);
            border-left:4px solid #2f81f7;border-radius:8px;padding:1rem 1.25rem;color:#79c0ff;
            margin-bottom:1.5rem;">
            <b>ℹ️ No audio dataset found — choose one of the 3 options below</b>
            </div>
            """, unsafe_allow_html=True)

            # ── OPTION 1: Small Real Dataset (RECOMMENDED) ──
            st.markdown("### ✅ Option 1 — Small Real Dataset (Recommended, ~200 MB)")
            st.markdown("""
            <div class="vc-card" style="border-color:#3fb950; border-width:2px;">
            <b style="color:#3fb950;">🏆 BEST CHOICE</b> — Real LJSpeech genuine audio + synthetic counterparts<br>
            <span style="color:#8b949e;">Downloads automatically. No manual steps needed.</span><br><br>
            <b>What you get:</b> Up to 1,000 genuine WAV files (real human speech) + matching synthetic clips<br>
            <b>Total disk usage:</b> ~200–400 MB &nbsp;|&nbsp; <b>Download time:</b> 5–15 mins on average connection<br>
            <b>Result:</b> Real, meaningful model performance on actual speech data
            </div>
            """, unsafe_allow_html=True)

            col_dl, col_info = st.columns([1, 1])
            with col_dl:
                n_files = st.slider("Files per class", 100, 1000, 500, 100,
                                    help="More files = better accuracy but longer download")
                if st.button("⬇️ Auto-Download Small Dataset", use_container_width=True, type="primary"):
                    progress_box = st.empty()
                    def _dl_cb(msg): progress_box.info(msg)
                    with st.spinner("Downloading LJSpeech subset …"):
                        from src.download_dataset import download_small_dataset
                        import os
                        root = download_small_dataset(n_genuine=n_files, n_synth=n_files, callback=_dl_cb)
                    if root is None:
                        st.error("Download failed. Check your internet connection.")
                    else:
                        os.environ["WAVEFAKE_ROOT"] = str(root)
                        with st.spinner("Building manifest …"):
                            from src.dataset_loader import build_manifest
                            df = build_manifest(force=True)
                        if df.empty:
                            st.error("Manifest build failed.")
                        else:
                            st.success(f"Dataset ready — {len(df)} audio files!")
                            st.cache_data.clear()
                            st.rerun()
            with col_info:
                st.markdown("""
                **Dataset contents:**
                - Genuine: Real LJSpeech recordings (audiobook-quality human speech)
                - Synthetic: TTS/vocoder-simulated clips
                - Split: 80% train / 20% test

                **Expected model accuracy (500 files/class):**
                - Accuracy: ~85–95%
                - F1-Score: ~0.85–0.95
                """)

            st.markdown("---")

            # ── OPTION 2: Instant Demo ──
            st.markdown("### ⚡ Option 2 — Instant Demo (No Download, 0 MB)")
            col_demo, col_demo_info = st.columns([1, 1])
            with col_demo:
                st.markdown("""
                <div class="vc-card">
                Generates 400 acoustic-simulation clips instantly (no internet needed).<br>
                <span style="color:#8b949e;">Good for testing the UI. Metrics won't reflect real-world performance.</span>
                </div>
                """, unsafe_allow_html=True)
                if st.button("⚡ Generate Demo Data & Build Manifest", use_container_width=True):
                    with st.spinner("Generating demo audio clips …"):
                        from src.demo_data import generate_demo_dataset
                        import os
                        demo_root = generate_demo_dataset(n_samples=200, force=True)
                        os.environ["WAVEFAKE_ROOT"] = str(demo_root)
                    with st.spinner("Building manifest …"):
                        from src.dataset_loader import build_manifest
                        df = build_manifest(force=True)
                    if df.empty:
                        st.error("Manifest build failed.")
                    else:
                        st.success(f"Demo manifest built — {len(df)} files ready!")
                        st.cache_data.clear()
                        st.rerun()
            with col_demo_info:
                st.markdown("""
                - 200 simulated genuine clips
                - 200 simulated synthetic clips
                - Ready in ~30 seconds, ~5 MB disk space
                """)

            st.markdown("---")

            # ── OPTION 3: Full WaveFake ──
            with st.expander("📦 Option 3 — Full WaveFake Dataset (72 GB, research-grade)"):
                st.markdown("""
                1. Visit **[WaveFake on GitHub](https://github.com/RUB-SysSec/WaveFake)**
                2. Download from **Zenodo** (~72 GB)
                3. Extract to `data/WaveFake/`
                4. Click Build Manifest below
                """)
                if st.button("🔄 Build Manifest (Full WaveFake)", use_container_width=True):
                    with st.spinner("Scanning dataset …"):
                        from src.dataset_loader import build_manifest
                        df = build_manifest(force=True)
                    if df.empty:
                        st.error("Could not build manifest.")
                    else:
                        st.success(f"Manifest built — {len(df)} records!")
                        st.cache_data.clear()
                        st.rerun()
        else:
            st.success(f"Dataset directory found at: `{WAVEFAKE_ROOT}`")
            if st.button("🔄 Build Manifest Now", type="primary"):
                with st.spinner("Scanning dataset …"):
                    from src.dataset_loader import build_manifest
                    df = build_manifest(force=True)
                if df.empty:
                    st.error("Could not build manifest.")
                else:
                    st.success(f"Manifest built — {len(df)} records!")
                    st.cache_data.clear()
                    st.rerun()
        st.stop()

    valid   = manifest[manifest["valid"] == True]
    invalid = manifest[manifest["valid"] != True]

    # ── KPI row ──────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Files",     len(manifest))
    k2.metric("Valid Files",     int(manifest["valid"].sum()))
    k3.metric("Invalid / Error", len(invalid))
    k4.metric("Genuine",         int((valid["label"] == 0).sum()))
    k5.metric("Synthetic",       int((valid["label"] == 1).sum()))

    col_l, col_r = st.columns(2)

    # ── Class distribution pie ────────────────
    with col_l:
        st.markdown("#### Class Distribution")
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#1c2128")
        counts  = [int((valid["label"] == 0).sum()), int((valid["label"] == 1).sum())]
        colors  = ["#3fb950", "#f85149"]
        wedges, texts, autotexts = ax.pie(
            counts, labels=["Genuine", "Synthetic"],
            autopct="%1.1f%%", colors=colors,
            textprops={"color": "#e6edf3"},
            wedgeprops={"linewidth": 2, "edgecolor": "#1c2128"},
        )
        for at in autotexts: at.set_fontsize(11)
        ax.set_title("Label Balance", color="#e6edf3")
        fig.patch.set_facecolor("#1c2128")
        st.pyplot(fig); plt.close(fig)

    # ── Vocoder distribution ──────────────────
    with col_r:
        st.markdown("#### Samples per Vocoder")
        voc_counts = valid.groupby("vocoder").size().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#1c2128")
        bars = ax.barh(voc_counts.index, voc_counts.values,
                       color=sns.color_palette("viridis", len(voc_counts)))
        ax.set_xlabel("Files", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.set_facecolor("#1c2128")
        fig.patch.set_facecolor("#1c2128")
        ax.spines[:].set_color("#30363d")
        st.pyplot(fig); plt.close(fig)

    # ── Duration stats ────────────────────────
    if "duration_s" in valid.columns:
        st.markdown("#### Audio Duration Statistics")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Mean Duration",   f"{valid['duration_s'].mean():.2f}s")
        d2.metric("Median Duration", f"{valid['duration_s'].median():.2f}s")
        d3.metric("Min Duration",    f"{valid['duration_s'].min():.2f}s")
        d4.metric("Max Duration",    f"{valid['duration_s'].max():.2f}s")

        fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#1c2128")
        for lbl, col in [(0, "#3fb950"), (1, "#f85149")]:
            subset = valid[valid["label"] == lbl]["duration_s"].dropna()
            ax.hist(subset, bins=40, alpha=0.65, color=col,
                    label=LABEL_MAP[lbl], edgecolor="none")
        ax.set_xlabel("Duration (s)", color="#8b949e")
        ax.set_ylabel("Count", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.set_facecolor("#1c2128"); fig.patch.set_facecolor("#1c2128")
        ax.spines[:].set_color("#30363d")
        ax.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="#e6edf3")
        ax.set_title("Duration Distribution by Class", color="#e6edf3")
        st.pyplot(fig); plt.close(fig)

    # ── Train / test split ────────────────────
    st.markdown("#### Train / Test Split")
    sp1, sp2 = st.columns(2)
    sp1.metric("Training samples", int((valid["split"] == "train").sum()))
    sp2.metric("Test samples",     int((valid["split"] == "test").sum()))

    # ── Invalid file report ───────────────────
    if len(invalid) > 0:
        with st.expander(f"⚠️ {len(invalid)} Invalid / Missing Files"):
            st.dataframe(invalid[["path", "error"]].head(50), use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 3 — AUDIO ANALYSIS
# ═════════════════════════════════════════════

elif page == "🎵 Audio Analysis":
    st.markdown('<div class="hero"><h1>🎵 Audio Analysis</h1><p>Waveform · Spectrogram · MFCC · Acoustic Features</p></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload an audio file for visual analysis",
        type=["wav", "flac", "mp3", "ogg"],
        key="audio_analysis_uploader",
    )

    if not uploaded:
        st.info("Upload a .wav / .flac / .mp3 / .ogg file to explore its acoustic features.")
        st.stop()

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        import librosa
        import librosa.display as ld

        y, sr = librosa.load(str(tmp_path), sr=SAMPLE_RATE, mono=True, duration=10.0)

        info_col, _ = st.columns([1, 2])
        with info_col:
            st.markdown(f"""
            <div class="vc-card">
            <b>Audio Information</b><br><br>
            <span class="mono">File</span>: {uploaded.name}<br>
            <span class="mono">Duration</span>: {len(y)/sr:.2f}s<br>
            <span class="mono">Sample Rate</span>: {sr} Hz<br>
            <span class="mono">Samples</span>: {len(y):,}<br>
            </div>""", unsafe_allow_html=True)

        # ── Waveform ──────────────────────────
        st.markdown("#### Waveform")
        fig, ax = plt.subplots(figsize=(10, 2.2), facecolor="#1c2128")
        time_axis = np.linspace(0, len(y)/sr, len(y))
        ax.plot(time_axis, y, color="#2f81f7", lw=0.5)
        ax.fill_between(time_axis, y, alpha=0.2, color="#2f81f7")
        ax.set_xlabel("Time (s)", color="#8b949e"); ax.set_ylabel("Amplitude", color="#8b949e")
        ax.tick_params(colors="#8b949e"); ax.set_facecolor("#1c2128")
        fig.patch.set_facecolor("#1c2128"); ax.spines[:].set_color("#30363d")
        ax.set_title("Waveform", color="#e6edf3")
        st.pyplot(fig); plt.close(fig)

        # ── Spectrogram ───────────────────────
        st.markdown("#### Mel Spectrogram")
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1c2128")
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = ld.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH,
                          x_axis="time", y_axis="mel", ax=ax, cmap="inferno")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel Spectrogram", color="#e6edf3")
        ax.tick_params(colors="#8b949e"); ax.set_facecolor("#1c2128")
        fig.patch.set_facecolor("#1c2128")
        st.pyplot(fig); plt.close(fig)

        # ── MFCC ──────────────────────────────
        st.markdown(f"#### MFCC ({N_MFCC} coefficients)")
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1c2128")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        img = ld.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH,
                          x_axis="time", ax=ax, cmap="coolwarm")
        fig.colorbar(img, ax=ax)
        ax.set_ylabel("MFCC Coefficient", color="#8b949e")
        ax.set_title("MFCC Heatmap", color="#e6edf3")
        ax.tick_params(colors="#8b949e"); ax.set_facecolor("#1c2128")
        fig.patch.set_facecolor("#1c2128")
        st.pyplot(fig); plt.close(fig)

        # ── Extracted feature table ───────────
        st.markdown("#### Extracted Acoustic Features")
        from src.audio_features import extract_features
        feat = extract_features(tmp_path)
        if feat is not None:
            feat_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Value"  : [round(float(v), 5) for v in feat],
            })
            st.dataframe(feat_df, use_container_width=True, height=300)
        else:
            st.warning("Feature extraction failed for this file.")

    except ImportError:
        st.error("librosa not installed. Run: `pip install librosa`")
    except Exception as exc:
        st.error(f"Error analysing audio: {exc}")
    finally:
        try: tmp_path.unlink()
        except: pass


# ═════════════════════════════════════════════
# PAGE 4 — MODEL TRAINING
# ═════════════════════════════════════════════

elif page == "🤖 Model Training":
    st.markdown('<div class="hero"><h1>🤖 Model Training</h1><p>Train the Random Forest Classifier on WaveFake features</p></div>', unsafe_allow_html=True)

    # Model config panel
    st.markdown("### ⚙️ Model Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        n_estimators = st.number_input("n_estimators (trees)", 50, 1000, 200, 50)
        max_features = st.selectbox("max_features", ["sqrt", "log2", None], index=0)
    with cfg_col2:
        max_depth = st.number_input("max_depth (0 = unlimited)", 0, 100, 0)
        max_depth = None if max_depth == 0 else max_depth
        min_samples_split = st.number_input("min_samples_split", 2, 50, 5)
    with cfg_col3:
        min_samples_leaf = st.number_input("min_samples_leaf", 1, 50, 2)
        class_weight = st.selectbox("class_weight", ["balanced", None], index=0)

    custom_params = {
        "n_estimators"     : n_estimators,
        "max_depth"        : max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf" : min_samples_leaf,
        "max_features"     : max_features,
        "class_weight"     : class_weight,
    }

    st.markdown("---")

    # Dataset check
    manifest = _load_manifest_cached()
    if manifest is None:
        st.error("Dataset manifest not found. Go to **Dataset Analysis** page to build it first.")
        st.stop()

    valid = manifest[manifest["valid"] == True]
    st.info(f"Dataset ready: **{len(valid)}** valid files "
            f"({int((valid['label']==0).sum())} genuine, {int((valid['label']==1).sum())} synthetic)")

    # Training button
    if st.button("🚀 Train Random Forest Model", use_container_width=True):
        progress_placeholder = st.empty()
        log_box              = st.empty()
        logs = []

        def _cb(msg):
            logs.append(msg)
            log_box.markdown(
                '<div class="vc-card"><pre style="color:#8b949e; font-size:0.8rem;">'
                + "\n".join(logs[-15:])
                + "</pre></div>",
                unsafe_allow_html=True,
            )

        with st.spinner("Training in progress …"):
            from src.train_model import train
            result = train(rf_params=custom_params, progress_callback=_cb)

        if result["success"]:
            st.success("✅ Training complete!")
            st.balloons()

            metrics = result.get("metrics", {})
            if metrics:
                st.markdown("### 📊 Evaluation Results")
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Accuracy",  f"{metrics.get('accuracy',  0):.4f}")
                r2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                r3.metric("Recall",    f"{metrics.get('recall',    0):.4f}")
                r4.metric("F1-Score",  f"{metrics.get('f1_score',  0):.4f}")
                r5.metric("ROC-AUC",   f"{metrics.get('roc_auc',   0) or 0:.4f}")

            # Confusion matrix
            if CONFUSION_MATRIX_PATH.exists():
                st.markdown("#### Confusion Matrix")
                st.image(str(CONFUSION_MATRIX_PATH))

            # Classification report
            if metrics.get("classification_report"):
                st.markdown("#### Classification Report")
                rpt = metrics["classification_report"]
                rpt_df = pd.DataFrame(rpt).T
                st.dataframe(rpt_df.style.format("{:.4f}", na_rep=""), use_container_width=True)

            # Feature importance
            if FEATURE_IMP_PATH.exists():
                st.markdown("#### Feature Importance")
                st.image(str(FEATURE_IMP_PATH))

            t_s = result.get("training_time_s", 0)
            st.info(f"⏱ Training time: **{t_s:.2f}s**")
        else:
            st.error(f"Training failed: {result.get('message', 'Unknown error')}")

    # Show existing model info
    elif MODEL_PATH.exists():
        st.markdown("### 📦 Currently Trained Model")
        cfg = _load_feature_cfg_cached()
        if cfg:
            st.markdown(f"""
            <div class="vc-card">
            <b>Model Details</b><br><br>
            <span class="mono">Algorithm</span>: Random Forest Classifier<br>
            <span class="mono">Training samples</span>: {cfg.get('train_samples', 'N/A')}<br>
            <span class="mono">Test samples</span>: {cfg.get('test_samples', 'N/A')}<br>
            <span class="mono">Feature dimensions</span>: {cfg.get('n_features', 'N/A')}<br>
            <span class="mono">Training time</span>: {cfg.get('training_time_s', 0):.2f}s<br>
            </div>
            """, unsafe_allow_html=True)

        metrics = _load_metrics_cached()
        if metrics:
            st.markdown("### 📈 Existing Evaluation Results")
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Accuracy",  f"{metrics.get('accuracy',  0):.4f}")
            e2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            e3.metric("Recall",    f"{metrics.get('recall',    0):.4f}")
            e4.metric("F1-Score",  f"{metrics.get('f1_score',  0):.4f}")
            e5.metric("ROC-AUC",   f"{metrics.get('roc_auc',   0) or 0:.4f}")

        if CONFUSION_MATRIX_PATH.exists(): st.image(str(CONFUSION_MATRIX_PATH))
        if FEATURE_IMP_PATH.exists():      st.image(str(FEATURE_IMP_PATH))


# ═════════════════════════════════════════════
# PAGE 5 — VOICE CLONING DETECTION
# ═════════════════════════════════════════════

elif page == "🛡️ Voice Cloning Detection":
    st.markdown('<div class="hero"><h1>🛡️ Voice Cloning Detection</h1><p>Upload audio · AI analyses acoustic features · Random Forest classifies</p></div>', unsafe_allow_html=True)

    from src.train_model import model_is_trained
    if not model_is_trained():
        st.markdown("""
        <div class="alert-danger">
        <b>Model Not Trained</b><br>
        The Random Forest model has not been trained yet.
        Go to the <b>🤖 Model Training</b> page and click <b>Train Model</b>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Threshold control
    with st.expander("⚙️ Detection Settings"):
        threshold = st.slider(
            "Spoof Probability Threshold",
            min_value=0.1, max_value=0.9, value=SPOOF_THRESHOLD, step=0.05,
            help="Probability ≥ threshold → classified as Synthetic / Suspected Spoof",
        )
        st.caption(f"Current threshold: **{threshold:.2f}** — lower values = more sensitive")

    uploaded = st.file_uploader(
        "Upload audio file(s) for detection",
        type=["wav", "flac", "mp3", "ogg"],
        accept_multiple_files=True,
        key="detection_uploader",
    )

    if not uploaded:
        st.info("Upload one or more audio files to analyse for voice cloning.")
        st.stop()

    from src.predict import predict

    for up_file in uploaded:
        with tempfile.NamedTemporaryFile(suffix=Path(up_file.name).suffix, delete=False) as tmp:
            tmp.write(up_file.read())
            tmp_path = Path(tmp.name)

        st.markdown(f"---\n### 🔍 Analysing: `{up_file.name}`")

        with st.spinner(f"Extracting features and classifying …"):
            result = predict(tmp_path, threshold=threshold)

        try: tmp_path.unlink()
        except: pass

        if not result["success"]:
            st.markdown(f'<div class="alert-danger">❌ {result["error"]}</div>', unsafe_allow_html=True)
            continue

        # Audio info
        info = result["audio_info"]
        icol1, icol2, icol3, icol4 = st.columns(4)
        icol1.metric("Format",   info.get("format", "?"))
        icol2.metric("Duration", f"{info.get('duration_s', 0):.2f}s")
        icol3.metric("Sample Rate", f"{info.get('sample_rate', 0)} Hz")
        icol4.metric("Channels", info.get("channels", "?"))

        # Verdict
        spoof_prob   = result["spoof_probability"]
        genuine_prob = result["genuine_probability"]
        is_spoof     = result["is_suspicious"]

        verdict_color = "#f85149" if is_spoof else "#3fb950"
        verdict_icon  = "🚨" if is_spoof else "✅"
        verdict_label = result["label_name"]

        st.markdown(f"""
        <div class="vc-card" style="border-color:{verdict_color}; border-width:2px;">
            <div style="display:flex; align-items:center; gap:1rem;">
                <div style="font-size:3rem;">{verdict_icon}</div>
                <div>
                    <div style="font-size:1.4rem; font-weight:800; color:{verdict_color};">
                        {verdict_label}
                    </div>
                    <div style="color:#8b949e; font-size:0.9rem;">
                        Model confidence score (spoof probability)
                    </div>
                </div>
            </div>
            <br>
            <b style="color:#8b949e;">Genuine Probability</b>
            {_render_prob_bar(genuine_prob, '#3fb950')}
            <br>
            <b style="color:#8b949e;">Synthetic Probability</b>
            {_render_prob_bar(spoof_prob, '#f85149')}
        </div>
        """, unsafe_allow_html=True)

        # Warning
        if is_spoof:
            st.markdown(f"""
            <div class="alert-danger">
            <b>⚠️ CYBERSECURITY WARNING</b><br><br>
            This recording has a <b>{spoof_prob*100:.1f}%</b> probability of being
            AI-generated or cloned voice audio.<br><br>
            <b>Recommendations:</b>
            <ul>
              <li>Do NOT authorise financial transactions or system access based solely on this voice.</li>
              <li>Perform independent identity verification (call back on a known number, use MFA).</li>
              <li>Report suspected voice-cloning attacks to your security team.</li>
              <li>This tool is a probabilistic detector — false negatives are possible.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-success">
            ✅ <b>Likely Genuine Speech</b> — spoof probability {spoof_prob*100:.1f}%
            (below threshold of {threshold*100:.0f}%).<br>
            <i>Note: No detection system is perfect. Maintain standard verification protocols.</i>
            </div>
            """, unsafe_allow_html=True)

        # Feature breakdown
        with st.expander("🔬 Extracted Feature Vector"):
            feat_dict = result.get("features", {})
            if feat_dict:
                feat_df = pd.DataFrame(
                    list(feat_dict.items()), columns=["Feature", "Value"]
                )
                st.dataframe(feat_df, use_container_width=True, height=300)

    # Prediction history
    from src.predict import load_history
    history = load_history()
    if history is not None and len(history) > 0:
        with st.expander(f"📋 Prediction History ({len(history)} entries)"):
            st.dataframe(history.tail(50), use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 6 — EVALUATION RESULTS
# ═════════════════════════════════════════════

elif page == "📈 Evaluation Results":
    st.markdown('<div class="hero"><h1>📈 Evaluation Results</h1><p>Full metrics, confusion matrix, ROC curve, and feature importance</p></div>', unsafe_allow_html=True)

    metrics = _load_metrics_cached()
    cfg     = _load_feature_cfg_cached()

    if metrics is None:
        st.warning("No evaluation results found. Train the model first.")
        st.stop()

    # ── KPI metrics ───────────────────────────
    st.markdown("### 📊 Performance Metrics (Held-Out Test Set)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy",  f"{metrics.get('accuracy',  0):.4f}")
    k2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    k3.metric("Recall",    f"{metrics.get('recall',    0):.4f}")
    k4.metric("F1-Score",  f"{metrics.get('f1_score',  0):.4f}")
    k5.metric("ROC-AUC",   f"{metrics.get('roc_auc',   0) or 0:.4f}")

    t1, t2 = st.columns(2)
    t1.metric("Test Samples",   metrics.get("n_test_samples", "N/A"))
    t2.metric("Inference Time", f"{metrics.get('inference_time_s', 0):.4f}s")

    if cfg:
        t3, t4 = st.columns(2)
        t3.metric("Training Samples",  cfg.get("train_samples", "N/A"))
        t4.metric("Training Time",     f"{cfg.get('training_time_s', 0):.2f}s")

    # ── Plots ─────────────────────────────────
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if CONFUSION_MATRIX_PATH.exists():
            st.markdown("#### Confusion Matrix")
            st.image(str(CONFUSION_MATRIX_PATH))
    with pcol2:
        if ROC_CURVE_PATH.exists():
            st.markdown("#### ROC Curve")
            st.image(str(ROC_CURVE_PATH))

    if FEATURE_IMP_PATH.exists():
        st.markdown("#### Top Feature Importances")
        st.image(str(FEATURE_IMP_PATH), use_container_width=True)

    # ── Classification report ─────────────────
    rpt = metrics.get("classification_report")
    if rpt:
        st.markdown("#### Classification Report")
        try:
            rpt_df = pd.DataFrame(rpt).T
            st.dataframe(
                rpt_df.style.format("{:.4f}", na_rep=""),
                use_container_width=True,
            )
        except Exception:
            st.json(rpt)

    # ── Confusion matrix raw ──────────────────
    cm = metrics.get("confusion_matrix")
    if cm:
        st.markdown("#### Confusion Matrix (Raw)")
        cm_df = pd.DataFrame(
            cm,
            index   = [f"Actual {LABEL_MAP[i]}" for i in range(len(cm))],
            columns = [f"Predicted {LABEL_MAP[i]}" for i in range(len(cm[0]))],
        )
        st.dataframe(cm_df, use_container_width=True)

    # ── Methodology note ──────────────────────
    st.markdown("""
    <div class="alert-info">
    <b>Evaluation Methodology</b><br>
    All metrics are computed on a <b>stratified held-out test set (20% of valid files)</b>
    that was never seen during training or scaler fitting.
    No data augmentation or post-hoc tuning was applied to improve test scores.
    Reported numbers reflect actual model performance on unseen data.
    </div>
    """, unsafe_allow_html=True)
