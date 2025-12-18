import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ================= CONFIG =================
st.set_page_config(
    page_title="Spam Classification System",
    layout="wide"
)

DATA_FILE = "spambase.csv"

# ================= DATA =================
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_FILE)
    if "class" not in df.columns:
        df.rename(columns={df.columns[-1]: "class"}, inplace=True)
    return df


@st.cache_data
def prepare_data(df):
    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_nb = scaler.fit_transform(X_train)
    X_test_nb = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_nb, X_test_nb, scaler


@st.cache_resource
def train_models(X_train, X_train_nb, y_train):
    nb = MultinomialNB()
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    nb.fit(X_train_nb, y_train)
    rf.fit(X_train, y_train)

    return nb, rf

# ================= FEATURE EXTRACTION =================
def extract_spambase_features(text, feature_columns):
    original_text = text
    lower_text = text.lower()

    words = re.findall(r"[a-zA-Z]+", lower_text)
    total_words = len(words) + 1

    features = {}

    for col in feature_columns:
        if col.startswith("word_freq_"):
            word = col.replace("word_freq_", "")
            features[col] = words.count(word) / total_words

        elif col.startswith("char_freq_"):
            ch = col.replace("char_freq_", "")
            features[col] = original_text.count(ch) / max(len(original_text), 1)

        elif col == "capital_run_length_average":
            runs = re.findall(r"[A-Z]+", original_text)
            features[col] = np.mean([len(r) for r in runs]) if runs else 0

        elif col == "capital_run_length_longest":
            runs = re.findall(r"[A-Z]+", original_text)
            features[col] = max([len(r) for r in runs]) if runs else 0

        elif col == "capital_run_length_total":
            runs = re.findall(r"[A-Z]+", original_text)
            features[col] = sum(len(r) for r in runs)

        else:
            features[col] = 0.0

    return pd.DataFrame([features])

# ================= HELPERS =================
def evaluate(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)

# ================= SESSION STATE =================
if "page" not in st.session_state:
    st.session_state.page = "Simulation"

# ===================================================
# ================= SIMULATION PAGE =================
# ===================================================
if st.session_state.page == "Simulation":
    st.title("📨 Email Spam Detection Simulation")

    df = load_dataset()
    feature_cols = df.drop(columns=["class"]).columns.tolist()

    (
        X_train, X_test, y_train, y_test,
        X_train_nb, X_test_nb, scaler
    ) = prepare_data(df)

    nb, rf = train_models(X_train, X_train_nb, y_train)

    best_model = rf if f1_score(
        y_test, rf.predict(X_test)
    ) >= f1_score(
        y_test, nb.predict(X_test_nb)
    ) else nb

    st.info(
        f"Using model: "
        f"{'Random Forest (Modern)' if best_model == rf else 'Naive Bayes (Baseline)'}"
    )

    uploaded = st.file_uploader(
        "Upload email file (.txt or .csv)",
        type=["txt", "csv"]
    )

    if uploaded:
        if uploaded.name.endswith(".txt"):
            text = uploaded.read().decode("utf-8", errors="ignore")
            st.text_area("Email Content", text, height=250)

            email_df = extract_spambase_features(text, feature_cols)

            if best_model == nb:
                email_df = scaler.transform(email_df)

        else:
            preview_df = pd.read_csv(uploaded)
            st.dataframe(preview_df.head())
            email_df = preview_df[feature_cols]

        prediction = best_model.predict(email_df)[0]

        if prediction == 1:
            st.error("🚨 SPAM DETECTED — This email is unsafe.")
        else:
            st.success("✅ EMAIL IS SAFE — No spam indicators detected.")

    st.divider()

    # 🔽 BUTTON AT THE BOTTOM
    if st.button("📊 View Training & Calculations"):
        st.session_state.page = "Training"
        st.rerun()

# ===================================================
# ================= TRAINING PAGE ===================
# ===================================================
if st.session_state.page == "Training":
    st.title("📊 Training & Model Evaluation")

    df = load_dataset()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    (
        X_train, X_test, y_train, y_test,
        X_train_nb, X_test_nb, scaler
    ) = prepare_data(df)

    nb, rf = train_models(X_train, X_train_nb, y_train)

    nb_pred = nb.predict(X_test_nb)
    rf_pred = rf.predict(X_test)

    results_df = pd.DataFrame([
        evaluate("Naive Bayes", y_test, nb_pred),
        evaluate("Random Forest", y_test, rf_pred)
    ])

    st.subheader("Performance Metrics")
    st.dataframe(results_df)

    col1, col2 = st.columns(2)
    with col1:
        plot_confusion(y_test, nb_pred, "Naive Bayes")
    with col2:
        plot_confusion(y_test, rf_pred, "Random Forest")

    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb.predict_proba(X_test_nb)[:, 1])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

    fig, ax = plt.subplots()
    ax.plot(fpr_nb, tpr_nb, label=f"NB AUC={auc(fpr_nb, tpr_nb):.2f}")
    ax.plot(fpr_rf, tpr_rf, label=f"RF AUC={auc(fpr_rf, tpr_rf):.2f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    if st.button("⬅ Back to Simulation"):
        st.session_state.page = "Simulation"
        st.rerun()
