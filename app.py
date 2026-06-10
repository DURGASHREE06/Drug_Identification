Content is user-generated and unverified.
"""
╔══════════════════════════════════════════════════════════════════════╗
║   AI-Based Drug Classification System — Streamlit UI                ║
║   Capstone Project | 2nd Year Engineering                           ║
║                                                                      ║
║   Run with:  streamlit run app.py                                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import json
import os
import io
import time

warnings.filterwarnings("ignore")

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Drug Classification AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { color: #60a5fa; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 3px solid #3b82f6;
        padding-left: 10px;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Prediction box */
    .pred-illicit {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .pred-nonillicit {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .pred-title { font-size: 1rem; color: #cbd5e1; margin-bottom: 6px; }
    .pred-label { font-size: 2.2rem; font-weight: 700; letter-spacing: 2px; }
    .pred-prob  { font-size: 1rem; color: #cbd5e1; margin-top: 6px; }

    /* Model prob bars */
    .model-bar-wrap { margin: 8px 0; }
    .model-bar-label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 3px; }

    /* Info box */
    .info-box {
        background: #1e293b;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #94a3b8;
        margin: 8px 0;
    }

    /* Step badge */
    .step-badge {
        display: inline-block;
        background: #1d4ed8;
        color: white;
        border-radius: 50%;
        width: 26px; height: 26px;
        text-align: center;
        line-height: 26px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #3b82f6 !important; }

    /* Buttons */
    .stButton > button {
        background: #1d4ed8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #2563eb; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #3b82f6;
        border-radius: 10px;
        padding: 8px;
    }

    /* Dataframe */
    .dataframe { font-size: 0.82rem; }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Navigation & Info
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧬 Drug Classification AI")
    st.markdown("*Capstone Project — 2nd Year Engineering*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home & Setup",
         "📊 Data Explorer",
         "🤖 Train Models",
         "📈 Results & Metrics",
         "🔮 Predict New Subject",
         "🧠 AI Explainability (SHAP)",
         "📋 AI Clinical Report"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Pipeline Steps:**")
    steps = [
        ("1", "Upload Dataset"),
        ("2", "Explore Data"),
        ("3", "Train ML + DL"),
        ("4", "View Results"),
        ("5", "Predict Subject"),
        ("6", "Explain with SHAP"),
        ("7", "AI Report"),
    ]
    for num, label in steps:
        trained = st.session_state.get("models_trained", False)
        done = (num in ["1"] and st.session_state.get("data_loaded", False)) or \
               (num in ["2"] and st.session_state.get("data_loaded", False)) or \
               (num in ["3","4","5","6","7"] and trained)
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} `{num}` {label}")

    st.divider()
    st.markdown(
        "<div style='font-size:0.75rem;color:#64748b;'>"
        "Models: Random Forest · XGBoost · Deep Neural Network<br>"
        "XAI: SHAP · AI Report: Claude API"
        "</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — shared across pages
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "data_loaded": False,
        "df": None,
        "X_train_raw": None, "X_test_raw": None,
        "X_train_sc": None,  "X_test_sc": None,
        "y_train": None,     "y_test": None,
        "FEATURE_COLS": None,
        "scaler": None,
        "models_trained": False,
        "rf": None, "gb": None, "gb_name": "XGBoost",
        "dnn_model": None,
        "rf_res": None, "gb_res": None, "dnn_res": None,
        "dnn_train_losses": [], "dnn_val_losses": [],
        "shap_values": None, "shap_sample_idx": None,
        "ai_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_sklearn():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, roc_curve,
        confusion_matrix, classification_report,
    )
    return True

load_sklearn()

def evaluate_model(model, name, X_te, y_te, y_prob):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "name": name,
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall":    recall_score(y_te, y_pred, zero_division=0),
        "f1":        f1_score(y_te, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_te, y_prob),
        "cm":        confusion_matrix(y_te, y_pred),
        "y_prob":    y_prob,
        "y_pred":    y_pred,
    }

def metric_color(val):
    if val >= 0.95: return "#22c55e"
    if val >= 0.90: return "#f59e0b"
    return "#ef4444"


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & SETUP
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home & Setup":
    st.markdown("# 🧬 AI-Based Drug Classification System")
    st.markdown("*Capstone Project · 2nd Year Engineering · ML + Deep Learning + XAI*")
    st.divider()

    # Architecture overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🤖 Machine Learning")
        st.markdown("""
- **Random Forest** — 200 trees, balanced class weight
- **XGBoost** — Gradient boosting, hist method
- Stratified 5-fold cross-validation
- Feature importance analysis
        """)
    with col2:
        st.markdown("### 🧠 Deep Learning")
        st.markdown("""
- **PyTorch Neural Network** (DrugNet)
- 5 layers: 256 → 128 → 64 → 32 → 2
- Batch Normalization + Dropout
- Adam optimizer + LR scheduler
        """)
    with col3:
        st.markdown("### 🔍 AI Explainability")
        st.markdown("""
- **SHAP** TreeExplainer on XGBoost
- Beeswarm & bar importance plots
- **Claude AI** clinical report
- Ensemble voting predictor
        """)

    st.divider()

    # Step-by-step guide
    st.markdown("## 📖 How to Use — Step by Step")

    steps_guide = [
        ("🏠", "Home & Setup", "You are here. Read the overview and understand the pipeline."),
        ("📊", "Data Explorer", "Upload your CSV dataset. View class balance, feature stats, and correlation heatmap."),
        ("🤖", "Train Models", "Click 'Train All Models'. RF + XGBoost train instantly. DNN trains for 80 epochs with live loss curve."),
        ("📈", "Results & Metrics", "View Accuracy, Precision, Recall, F1, AUC-ROC for all 3 models side by side with confusion matrices."),
        ("🔮", "Predict New Subject", "Enter a subject's feature values manually. All models vote → Ensemble gives final ILLICIT / NON-ILLICIT result."),
        ("🧠", "AI Explainability", "SHAP explains *why* each prediction was made — which features pushed the model toward Illicit."),
        ("📋", "AI Clinical Report", "Enter your Anthropic API key → Claude generates a professional clinical risk report based on predictions."),
    ]

    for i, (icon, title, desc) in enumerate(steps_guide, 1):
        with st.container():
            c1, c2 = st.columns([0.07, 0.93])
            with c1:
                st.markdown(
                    f"<div style='background:#1d4ed8;color:white;border-radius:50%;"
                    f"width:34px;height:34px;text-align:center;line-height:34px;"
                    f"font-weight:700;font-size:1rem;'>{i}</div>",
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(f"**{icon} {title}** — {desc}")

    st.divider()

    # Install commands
    st.markdown("## ⚙️ Installation")
    st.code("""# Install all required packages
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
pip install xgboost torch torchvision shap anthropic joblib

# Run the app
streamlit run app.py""", language="bash")

    st.markdown(
        "<div class='info-box'>💡 <b>Tip:</b> If PyTorch or XGBoost is not installed, "
        "the app will automatically fall back to sklearn alternatives — "
        "you won't lose any functionality.</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Data Explorer":
    st.markdown("# 📊 Data Explorer")
    st.divider()

    uploaded = st.file_uploader(
        "Upload your dataset CSV",
        type=["csv"],
        help="Upload drug_cleaned_engineered.csv or any compatible dataset."
    )

    if uploaded:
        df = pd.read_csv(uploaded)

        DROP_COLS = [
            "ID", "Age_Label", "Gender_Label", "Education_Label",
            "Country_Label", "Ethnicity_Label",
            "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc",
            "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD",
            "Meth", "Mushrooms", "Nicotine", "Semer", "VSA",
            "Illicit_Total", "Illicit_Count", "Polydrug", "Is_Illicit",
        ]
        df["Is_Illicit"] = (df["Illicit_Count"] >= 1).astype(int)
        FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X = df[FEATURE_COLS].values
        y = df["Is_Illicit"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        X_train_sc, X_test_sc, _, _ = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42, stratify=y
        )

        # Save to session
        st.session_state.update({
            "data_loaded": True, "df": df,
            "FEATURE_COLS": FEATURE_COLS, "scaler": scaler,
            "X_train_raw": X_train_raw, "X_test_raw": X_test_raw,
            "X_train_sc": X_train_sc, "X_test_sc": X_test_sc,
            "y_train": y_train, "y_test": y_test,
        })

        # Summary metrics
        class_counts = df["Is_Illicit"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Features Used", len(FEATURE_COLS))
        c3.metric("Non-Illicit (0)", f"{class_counts.get(0,0):,}")
        c4.metric("Illicit (1)", f"{class_counts.get(1,0):,}")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Preview", "📊 Statistics", "🔥 Correlation", "🥧 Class Balance"]
        )

        with tab1:
            st.dataframe(df[FEATURE_COLS + ["Is_Illicit"]].head(50), use_container_width=True)

        with tab2:
            st.dataframe(
                df[FEATURE_COLS].describe().round(3),
                use_container_width=True
            )

        with tab3:
            numeric_cols = df[FEATURE_COLS].select_dtypes(include=np.number).columns.tolist()
            corr_cols = numeric_cols[:15]  # top 15 for readability
            fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
            fig_corr.patch.set_facecolor("#0f1117")
            ax_corr.set_facecolor("#1a1f2e")
            corr_matrix = df[corr_cols].corr()
            sns.heatmap(corr_matrix, ax=ax_corr, cmap="coolwarm",
                        linewidths=0.5, annot=len(corr_cols) <= 12,
                        fmt=".2f", annot_kws={"size": 8},
                        cbar_kws={"shrink": 0.8})
            ax_corr.set_title("Feature Correlation Matrix", color="white", pad=10)
            ax_corr.tick_params(colors="white", labelsize=8)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig_corr)

        with tab4:
            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            fig_pie.patch.set_facecolor("#0f1117")
            ax_pie.pie(
                class_counts.values,
                labels=["Non-Illicit", "Illicit"],
                colors=["#3b82f6", "#ef4444"],
                autopct="%1.1f%%",
                startangle=140,
                textprops={"color": "white", "fontsize": 11},
            )
            ax_pie.set_title("Class Distribution", color="white")
            st.pyplot(fig_pie)

        st.success("✅ Dataset loaded and ready. Go to **Train Models** next.")

    elif st.session_state.get("data_loaded"):
        st.info("✅ Dataset already loaded from earlier. Proceed to **Train Models**.")
    else:
        st.markdown(
            "<div class='info-box'>⬆️ Upload your <b>drug_cleaned_engineered.csv</b> file above to begin.</div>",
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Train Models":
    st.markdown("# 🤖 Train Models")
    st.divider()

    if not st.session_state.get("data_loaded"):
        st.warning("⚠️ Please upload your dataset in **Data Explorer** first.")
        st.stop()

    X_train_raw = st.session_state["X_train_raw"]
    X_test_raw  = st.session_state["X_test_raw"]
    X_train_sc  = st.session_state["X_train_sc"]
    X_test_sc   = st.session_state["X_test_sc"]
    y_train     = st.session_state["y_train"]
    y_test      = st.session_state["y_test"]
    FEATURE_COLS = st.session_state["FEATURE_COLS"]

    # Config columns
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⚙️ ML Model Settings")
        n_estimators = st.slider("Number of Trees / Estimators", 50, 500, 200, 50)
        max_depth_xgb = st.slider("XGBoost Max Depth", 2, 8, 4)
        lr_xgb = st.select_slider("XGBoost Learning Rate", [0.01, 0.05, 0.1, 0.2], value=0.05)
    with c2:
        st.markdown("### 🧠 DNN Settings")
        dnn_epochs = st.slider("DNN Training Epochs", 20, 150, 80, 10)
        dnn_lr = st.select_slider("DNN Learning Rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
        dnn_batch = st.select_slider("Batch Size", [32, 64, 128, 256], value=64)

    st.divider()

    if st.button("🚀 Train All Models"):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        # ── Random Forest ──────────────────────────────────────────────────
        with st.status("Training Random Forest…", expanded=True) as status_rf:
            st.write("Fitting 200 trees with balanced class weights…")
            rf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            rf.fit(X_train_raw, y_train)
            rf_prob = rf.predict_proba(X_test_raw)[:, 1]
            rf_res = evaluate_model(rf, "Random Forest", X_test_raw, y_test, rf_prob)
            st.session_state["rf"] = rf
            st.session_state["rf_res"] = rf_res
            status_rf.update(
                label=f"✅ Random Forest — AUC: {rf_res['auc']*100:.2f}%",
                state="complete"
            )

        # ── XGBoost ───────────────────────────────────────────────────────
        with st.status("Training XGBoost…", expanded=True) as status_gb:
            st.write("Fitting gradient boosting with histogram trees…")
            try:
                import xgboost as xgb
                gb = xgb.XGBClassifier(
                    n_estimators=n_estimators, learning_rate=lr_xgb,
                    max_depth=max_depth_xgb, subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    tree_method="hist", random_state=42,
                )
                gb_name = "XGBoost"
            except ImportError:
                gb = GradientBoostingClassifier(
                    n_estimators=n_estimators, learning_rate=lr_xgb,
                    max_depth=max_depth_xgb, subsample=0.8, random_state=42,
                )
                gb_name = "Gradient Boosting"
            gb.fit(X_train_raw, y_train)
            gb_prob = gb.predict_proba(X_test_raw)[:, 1]
            gb_res = evaluate_model(gb, gb_name, X_test_raw, y_test, gb_prob)
            st.session_state.update({"gb": gb, "gb_name": gb_name, "gb_res": gb_res})
            status_gb.update(
                label=f"✅ {gb_name} — AUC: {gb_res['auc']*100:.2f}%",
                state="complete"
            )

        # ── Deep Neural Network ────────────────────────────────────────────
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class DrugNet(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_dim, 256), nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.1), nn.Dropout(0.3),
                        nn.Linear(256, 128), nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.1), nn.Dropout(0.25),
                        nn.Linear(128, 64),  nn.BatchNorm1d(64),
                        nn.LeakyReLU(0.1), nn.Dropout(0.2),
                        nn.Linear(64, 32),   nn.LeakyReLU(0.1),
                        nn.Linear(32, 2),
                    )
                def forward(self, x): return self.network(x)

            dnn_model = DrugNet(input_dim=len(FEATURE_COLS)).to(DEVICE)
            X_tr_t = torch.FloatTensor(X_train_sc).to(DEVICE)
            y_tr_t  = torch.LongTensor(y_train).to(DEVICE)
            X_te_t  = torch.FloatTensor(X_test_sc).to(DEVICE)

            class_counts_arr = np.bincount(y_train)
            class_weights = torch.FloatTensor(
                [len(y_train)/(2*c) for c in class_counts_arr]
            ).to(DEVICE)
            criterion  = nn.CrossEntropyLoss(weight=class_weights)
            optimizer  = optim.Adam(dnn_model.parameters(), lr=dnn_lr, weight_decay=1e-4)
            scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            train_ds   = TensorDataset(X_tr_t, y_tr_t)
            train_loader = DataLoader(train_ds, batch_size=dnn_batch, shuffle=True)

            train_losses, val_losses = [], []
            best_auc, best_state = 0, None

            from sklearn.metrics import roc_auc_score

            progress_bar = st.progress(0, text="Training Deep Neural Network…")
            loss_chart   = st.empty()

            for epoch in range(dnn_epochs):
                dnn_model.train()
                epoch_loss = 0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(dnn_model(xb), yb)
                    loss.backward(); optimizer.step()
                    epoch_loss += loss.item()

                dnn_model.eval()
                with torch.no_grad():
                    out      = dnn_model(X_te_t)
                    val_loss = criterion(out, torch.LongTensor(y_test).to(DEVICE)).item()
                    probs_np = torch.softmax(out, dim=1)[:,1].cpu().numpy()
                    val_auc  = roc_auc_score(y_test, probs_np)

                avg_train = epoch_loss / len(train_loader)
                train_losses.append(avg_train)
                val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.clone() for k,v in dnn_model.state_dict().items()}

                pct = int((epoch+1)/dnn_epochs*100)
                progress_bar.progress(
                    pct,
                    text=f"Epoch {epoch+1}/{dnn_epochs} | Val AUC: {val_auc*100:.2f}%"
                )

                if (epoch+1) % 5 == 0:
                    fig_loss, ax_loss = plt.subplots(figsize=(7,3))
                    fig_loss.patch.set_facecolor("#0f1117")
                    ax_loss.set_facecolor("#1a1f2e")
                    ax_loss.plot(train_losses, color="#3b82f6", lw=1.8, label="Train Loss")
                    ax_loss.plot(val_losses,   color="#f472b6", lw=1.8, linestyle="--", label="Val Loss")
                    ax_loss.set_xlabel("Epoch", color="white")
                    ax_loss.set_ylabel("Loss", color="white")
                    ax_loss.tick_params(colors="white")
                    ax_loss.legend(facecolor="#1a1f2e", labelcolor="white")
                    ax_loss.set_title(f"DNN Training — Epoch {epoch+1}", color="white")
                    for spine in ax_loss.spines.values(): spine.set_edgecolor("#2d3748")
                    loss_chart.pyplot(fig_loss)
                    plt.close(fig_loss)

            dnn_model.load_state_dict(best_state)
            dnn_model.eval()
            with torch.no_grad():
                final_out  = dnn_model(X_te_t)
                dnn_probs  = torch.softmax(final_out, dim=1)[:,1].cpu().numpy()
            dnn_res = evaluate_model(None, "Deep Neural Network", X_test_sc, y_test, dnn_probs)

            st.session_state.update({
                "dnn_model": dnn_model, "dnn_res": dnn_res,
                "dnn_train_losses": train_losses, "dnn_val_losses": val_losses,
                "DEVICE": str(DEVICE),
            })
            st.success(f"✅ DNN done — Best Val AUC: {best_auc*100:.2f}%")

        except ImportError:
            st.warning("⚠️ PyTorch not installed — DNN skipped. `pip install torch`")

        st.session_state["models_trained"] = True
        st.success("🎉 All models trained! Go to **Results & Metrics** to view the dashboard.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS & METRICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Results & Metrics":
    st.markdown("# 📈 Results & Metrics")
    st.divider()

    if not st.session_state.get("models_trained"):
        st.warning("⚠️ Please train models first on the **Train Models** page.")
        st.stop()

    from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

    results = [r for r in [
        st.session_state.get("rf_res"),
        st.session_state.get("gb_res"),
        st.session_state.get("dnn_res"),
    ] if r is not None]

    y_test = st.session_state["y_test"]
    model_colors = ["#3b82f6", "#10b981", "#a855f7"]

    # ── Top metric cards ─────────────────────────────────────────────────────
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    for res, color in zip(results, model_colors):
        st.markdown(
            f"<div class='section-header' style='border-color:{color}'>"
            f"🤖 {res['name']}</div>",
            unsafe_allow_html=True
        )
        cols = st.columns(5)
        for col, key, label in zip(cols, metric_keys, metric_labels):
            val = res[key]
            col.metric(label, f"{val*100:.2f}%")

    st.divider()

    # ── Tabs: Comparison | ROC | Confusion Matrices | DNN Curve ─────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Comparison", "📉 ROC Curves", "🔲 Confusion Matrices", "📈 DNN Curve"]
    )

    with tab1:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        fig_bar.patch.set_facecolor("#0f1117")
        ax_bar.set_facecolor("#1a1f2e")

        x = np.arange(len(metric_labels))
        w = 0.25
        for i, (res, color) in enumerate(zip(results, model_colors)):
            vals = [res[k]*100 for k in metric_keys]
            bars = ax_bar.bar(x + i*w - w*(len(results)-1)/2, vals, w,
                              label=res["name"], color=color, alpha=0.85)
            ax_bar.bar_label(bars, fmt="%.1f%%", fontsize=7.5, padding=2, color="white")

        ax_bar.set_xticks(x); ax_bar.set_xticklabels(metric_labels, color="white")
        ax_bar.set_ylim(75, 110); ax_bar.set_ylabel("Score (%)", color="white")
        ax_bar.tick_params(colors="white")
        ax_bar.set_title("Performance Metrics — All Models", color="white")
        ax_bar.legend(facecolor="#1a1f2e", labelcolor="white")
        ax_bar.grid(axis="y", alpha=0.2, color="white")
        for spine in ax_bar.spines.values(): spine.set_edgecolor("#2d3748")
        st.pyplot(fig_bar)

        # Summary table
        rows = []
        for res in results:
            rows.append({
                "Model": res["name"],
                "Accuracy": f"{res['accuracy']*100:.2f}%",
                "Precision": f"{res['precision']*100:.2f}%",
                "Recall": f"{res['recall']*100:.2f}%",
                "F1 Score": f"{res['f1']*100:.2f}%",
                "AUC-ROC": f"{res['auc']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab2:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
        fig_roc.patch.set_facecolor("#0f1117")
        ax_roc.set_facecolor("#1a1f2e")
        for res, color in zip(results, model_colors):
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            ax_roc.plot(fpr, tpr, color=color, lw=2,
                        label=f"{res['name']} (AUC={res['auc']*100:.2f}%)")
        ax_roc.plot([0,1],[0,1], "w--", lw=1, alpha=0.4, label="Random baseline")
        ax_roc.set_xlabel("False Positive Rate", color="white")
        ax_roc.set_ylabel("True Positive Rate", color="white")
        ax_roc.set_title("ROC Curves", color="white")
        ax_roc.legend(facecolor="#1a1f2e", labelcolor="white", fontsize=9)
        ax_roc.tick_params(colors="white")
        ax_roc.grid(alpha=0.15, color="white")
        for spine in ax_roc.spines.values(): spine.set_edgecolor("#2d3748")
        st.pyplot(fig_roc)

    with tab3:
        cmaps = ["Blues", "Greens", "Purples"]
        cols_cm = st.columns(len(results))
        for col, res, cmap in zip(cols_cm, results, cmaps):
            with col:
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
                fig_cm.patch.set_facecolor("#0f1117")
                ConfusionMatrixDisplay(
                    res["cm"], display_labels=["Non-Illicit", "Illicit"]
                ).plot(ax=ax_cm, colorbar=False, cmap=cmap)
                ax_cm.set_title(res["name"], color="white", fontsize=9)
                ax_cm.tick_params(colors="white", labelsize=8)
                ax_cm.xaxis.label.set_color("white")
                ax_cm.yaxis.label.set_color("white")
                for spine in ax_cm.spines.values(): spine.set_edgecolor("#2d3748")
                st.pyplot(fig_cm)

    with tab4:
        train_losses = st.session_state.get("dnn_train_losses", [])
        val_losses   = st.session_state.get("dnn_val_losses", [])
        if train_losses:
            fig_curve, ax_curve = plt.subplots(figsize=(8, 4))
            fig_curve.patch.set_facecolor("#0f1117")
            ax_curve.set_facecolor("#1a1f2e")
            ax_curve.plot(train_losses, color="#3b82f6", lw=2, label="Train Loss")
            ax_curve.plot(val_losses, color="#f472b6", lw=2, linestyle="--", label="Val Loss")
            ax_curve.set_xlabel("Epoch", color="white")
            ax_curve.set_ylabel("Loss", color="white")
            ax_curve.set_title("DNN Training & Validation Loss", color="white")
            ax_curve.legend(facecolor="#1a1f2e", labelcolor="white")
            ax_curve.tick_params(colors="white")
            ax_curve.grid(alpha=0.15, color="white")
            for spine in ax_curve.spines.values(): spine.set_edgecolor("#2d3748")
            st.pyplot(fig_curve)
        else:
            st.info("DNN was not trained (PyTorch unavailable).")

    # Feature Importance
    st.divider()
    st.markdown("<div class='section-header'>🌲 Random Forest Feature Importance</div>", unsafe_allow_html=True)
    rf = st.session_state.get("rf")
    FEATURE_COLS = st.session_state.get("FEATURE_COLS")
    if rf is not None:
        fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
        fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
        fig_fi.patch.set_facecolor("#0f1117")
        ax_fi.set_facecolor("#1a1f2e")
        colors_fi = ["#3b82f6" if v > fi.mean() else "#64748b" for v in fi]
        fi.plot(kind="barh", ax=ax_fi, color=colors_fi)
        ax_fi.set_title("Top 15 Feature Importances — Random Forest", color="white")
        ax_fi.set_xlabel("Importance", color="white")
        ax_fi.tick_params(colors="white", labelsize=8)
        for i, v in enumerate(fi): ax_fi.text(v+0.001, i, f"{v*100:.2f}%", va="center", fontsize=8, color="white")
        ax_fi.grid(axis="x", alpha=0.2, color="white")
        for spine in ax_fi.spines.values(): spine.set_edgecolor("#2d3748")
        st.pyplot(fig_fi)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT NEW SUBJECT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Predict New Subject":
    st.markdown("# 🔮 Predict New Subject")
    st.divider()

    if not st.session_state.get("models_trained"):
        st.warning("⚠️ Please train models first.")
        st.stop()

    FEATURE_COLS = st.session_state["FEATURE_COLS"]
    scaler = st.session_state["scaler"]

    st.markdown("### Enter Subject Feature Values")
    st.markdown(
        "<div class='info-box'>Enter the subject's feature values below. "
        "Unknown fields default to 0. All models will vote and produce an ensemble result.</div>",
        unsafe_allow_html=True
    )

    # Dynamically build inputs from feature list
    input_vals = {}
    col_groups = [FEATURE_COLS[i:i+3] for i in range(0, len(FEATURE_COLS), 3)]
    for group in col_groups:
        cols = st.columns(3)
        for col, feat in zip(cols, group):
            with col:
                input_vals[feat] = st.number_input(
                    feat, value=0.0, step=0.1,
                    key=f"pred_{feat}"
                )

    st.divider()
    if st.button("🔮 Run Ensemble Prediction"):
        raw_arr = np.array([[input_vals.get(c, 0) for c in FEATURE_COLS]])
        sc_arr  = scaler.transform(raw_arr)

        probs = {}
        rf = st.session_state.get("rf")
        gb = st.session_state.get("gb")
        dnn_model = st.session_state.get("dnn_model")

        if rf:  probs["Random Forest"] = float(rf.predict_proba(raw_arr)[0][1])
        if gb:  probs[st.session_state["gb_name"]] = float(gb.predict_proba(raw_arr)[0][1])

        if dnn_model is not None:
            try:
                import torch
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dnn_model.eval()
                with torch.no_grad():
                    t   = torch.FloatTensor(sc_arr).to(DEVICE)
                    out = dnn_model(t)
                    p   = torch.softmax(out, dim=1).cpu().numpy()[0][1]
                probs["Deep Neural Network"] = float(p)
            except Exception:
                pass

        ensemble_prob = float(np.mean(list(probs.values()))) if probs else 0.5
        is_illicit    = ensemble_prob >= 0.5
        label         = "ILLICIT" if is_illicit else "NON-ILLICIT"
        css_class     = "pred-illicit" if is_illicit else "pred-nonillicit"
        label_color   = "#fca5a5" if is_illicit else "#86efac"

        # Result box
        st.markdown(
            f"<div class='{css_class}'>"
            f"<div class='pred-title'>Ensemble Prediction</div>"
            f"<div class='pred-label' style='color:{label_color};'>{label}</div>"
            f"<div class='pred-prob'>Confidence: {ensemble_prob*100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("### Individual Model Votes")
        for mname, p in probs.items():
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.progress(p, text=f"{mname}: {p*100:.1f}%")
            with col_r:
                vote = "🔴 ILLICIT" if p >= 0.5 else "🟢 NON-ILLICIT"
                st.markdown(f"**{vote}**")

        # Save for report
        st.session_state["last_prediction"] = {
            "input": input_vals, "label": label,
            "prob": ensemble_prob, "per_model": probs
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — AI EXPLAINABILITY (SHAP)
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 AI Explainability (SHAP)":
    st.markdown("# 🧠 AI Explainability (SHAP)")
    st.divider()

    if not st.session_state.get("models_trained"):
        st.warning("⚠️ Please train models first.")
        st.stop()

    st.markdown(
        "<div class='info-box'>"
        "<b>SHAP (SHapley Additive exPlanations)</b> explains <i>why</i> a model made each prediction. "
        "Each bar shows how much a feature pushed the model toward or away from classifying as Illicit."
        "</div>",
        unsafe_allow_html=True
    )

    try:
        import shap
        gb = st.session_state.get("gb")
        X_test_raw  = st.session_state["X_test_raw"]
        FEATURE_COLS = st.session_state["FEATURE_COLS"]

        sample_size = st.slider("Number of test samples to explain", 50, 300, 150, 50)

        if st.button("⚡ Compute SHAP Values"):
            with st.spinner("Computing SHAP values…"):
                explainer = shap.TreeExplainer(gb)
                idx = np.random.choice(len(X_test_raw), min(sample_size, len(X_test_raw)), replace=False)
                shap_values = explainer.shap_values(X_test_raw[idx])
                if isinstance(shap_values, list):
                    shap_vals_illicit = shap_values[1]
                else:
                    shap_vals_illicit = shap_values

                st.session_state["shap_values"] = shap_vals_illicit
                st.session_state["shap_idx"] = idx

            st.success("✅ SHAP values computed!")

        shap_vals = st.session_state.get("shap_values")
        shap_idx  = st.session_state.get("shap_idx")

        if shap_vals is not None:
            tab1, tab2 = st.tabs(["📊 Bar Summary", "🐝 Beeswarm Plot"])

            with tab1:
                mean_abs = np.abs(shap_vals).mean(axis=0)
                top_idx  = np.argsort(mean_abs)[-15:]
                top_feats = [FEATURE_COLS[i] for i in top_idx]
                top_vals  = mean_abs[top_idx]

                fig_s, ax_s = plt.subplots(figsize=(8, 5))
                fig_s.patch.set_facecolor("#0f1117")
                ax_s.set_facecolor("#1a1f2e")
                ax_s.barh(top_feats, top_vals, color="#f59e0b", alpha=0.85)
                ax_s.set_title("SHAP Feature Impact — Mean |SHAP Value|", color="white")
                ax_s.set_xlabel("|SHAP value|", color="white")
                ax_s.tick_params(colors="white", labelsize=9)
                ax_s.grid(axis="x", alpha=0.2, color="white")
                for spine in ax_s.spines.values(): spine.set_edgecolor("#2d3748")
                st.pyplot(fig_s)

            with tab2:
                fig_b, _ = plt.subplots(figsize=(9, 6))
                fig_b.patch.set_facecolor("#0f1117")
                shap.summary_plot(
                    shap_vals, X_test_raw[shap_idx],
                    feature_names=FEATURE_COLS,
                    show=False, plot_type="dot",
                )
                plt.title("SHAP Beeswarm — Illicit Drug Prediction", color="white")
                plt.gcf().patch.set_facecolor("#0f1117")
                st.pyplot(fig_b)

    except ImportError:
        st.error("SHAP not installed. Run: `pip install shap`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AI CLINICAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📋 AI Clinical Report":
    st.markdown("# 📋 AI Clinical Report")
    st.divider()

    st.markdown(
        "<div class='info-box'>"
        "This page uses <b>Claude (Anthropic API)</b> to generate a professional "
        "clinical risk report based on the model predictions and subject features. "
        "Run a prediction first on the <b>Predict New Subject</b> page."
        "</div>",
        unsafe_allow_html=True
    )

    api_key = st.text_input(
        "🔑 Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key from console.anthropic.com"
    )

    pred_data = st.session_state.get("last_prediction")

    if pred_data:
        st.markdown("### Last Prediction Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", pred_data["label"])
        c2.metric("Ensemble Confidence", f"{pred_data['prob']*100:.1f}%")
        c3.metric("Models Voted", len(pred_data["per_model"]))

        for mname, p in pred_data["per_model"].items():
            st.progress(p, text=f"{mname}: {p*100:.1f}%")
    else:
        st.warning("⚠️ No prediction yet. Go to **Predict New Subject** first.")

    st.divider()

    if st.button("📝 Generate AI Clinical Report") and pred_data:
        if not api_key:
            st.error("Please enter your Anthropic API key above.")
        else:
            with st.spinner("Generating clinical report with Claude…"):
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)

                    model_summary = "\n".join([
                        f"  - {m}: {p*100:.1f}% probability of illicit use"
                        for m, p in pred_data["per_model"].items()
                    ])
                    feature_summary = "\n".join([
                        f"  - {k}: {v}"
                        for k, v in list(pred_data["input"].items())[:10]
                        if v != 0
                    ]) or "  - All features at baseline (0)"

                    prompt = f"""You are a clinical AI drug risk assessment system.
A machine learning ensemble of {len(pred_data['per_model'])} models analyzed a subject's profile.

SUBJECT FEATURES (non-zero values):
{feature_summary}

MODEL PREDICTIONS:
{model_summary}

ENSEMBLE RESULT: {pred_data['label']} (Avg probability: {pred_data['prob']*100:.1f}%)

Write a concise professional clinical risk report (200-250 words) that:
1. States the risk classification and confidence
2. Highlights 2-3 most concerning features
3. Explains what the ML models detected
4. Recommends next steps for clinical follow-up
5. Notes AI limitations in drug classification

Use professional medical/clinical tone. Do not be alarmist."""

                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=512,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    report = message.content[0].text
                    st.session_state["ai_report"] = report

                except ImportError:
                    st.error("`anthropic` not installed. Run: `pip install anthropic`")
                    report = None
                except Exception as e:
                    st.error(f"API Error: {e}")
                    report = None

    report = st.session_state.get("ai_report")
    if report:
        st.markdown("### 🏥 Clinical Risk Report")
        st.markdown(
            f"<div style='background:#1e293b;border-left:3px solid #3b82f6;"
            f"border-radius:0 10px 10px 0;padding:20px 24px;"
            f"font-size:0.92rem;color:#e2e8f0;line-height:1.8;'>{report}</div>",
            unsafe_allow_html=True
        )
        buf = io.BytesIO()
        buf.write(f"AI CLINICAL RISK REPORT\n{'='*60}\n\n".encode())
        buf.write(f"Prediction: {pred_data['label']}\n".encode())
        buf.write(f"Confidence: {pred_data['prob']*100:.1f}%\n\n".encode())
        buf.write(f"{'='*60}\n\n{report}\n".encode())
        buf.seek(0)
        st.download_button(
            "⬇️ Download Report as TXT",
            data=buf,
            file_name="ai_clinical_report.txt",
            mime="text/plain"
        )
