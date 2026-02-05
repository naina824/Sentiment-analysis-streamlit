import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# LOAD MODELS & METRICS (SAFE)
# =====================================================
try:
    nb = pickle.load(open(os.path.join(MODELS_DIR, "nb_model.pkl"), "rb"))
    svm = pickle.load(open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb"))
    tfidf = pickle.load(open(os.path.join(MODELS_DIR, "tfidf.pkl"), "rb"))
    metrics = pickle.load(open(os.path.join(MODELS_DIR, "metrics.pkl"), "rb"))
except Exception:
    st.error("❌ Models not found. Please run the training script first.")
    st.stop()

# Safe access
nb_acc = metrics.get("nb_accuracy", 0.0)
svm_acc = metrics.get("svm_accuracy", 0.0)
cm_nb = metrics.get("cm_nb")
cm_svm = metrics.get("cm_svm")
nb_report = metrics.get("nb_report")
svm_report = metrics.get("svm_report")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio(
    "Choose a section",
    ["🏠 Live Demo", "📊 Model Performance", "ℹ️ About Project"]
)

st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Python & Streamlit")

# =====================================================
# SECTION 1: LIVE DEMO
# =====================================================
if section == "🏠 Live Demo":
    st.title("🔍 Live Sentiment Prediction")
    st.write(
        "Enter a sentence below and see predictions from both "
        "**Naive Bayes** and **SVM** models."
    )

    user_input = st.text_area(
        "Enter your sentence here:",
        height=120,
        placeholder="Example: I absolutely loved this movie!"
    )

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            vec = tfidf.transform([user_input])
            nb_pred = nb.predict(vec)[0]
            svm_pred = svm.predict(vec)[0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Naive Bayes")
                if nb_pred == 1:
                    st.success("🟢 Positive")
                else:
                    st.error("🔴 Negative")

            with col2:
                st.subheader("SVM (LinearSVC)")
                if svm_pred == 1:
                    st.success("🟢 Positive")
                else:
                    st.error("🔴 Negative")


# =====================================================
# SECTION 2: MODEL PERFORMANCE
# =====================================================
elif section == "📊 Model Performance":
    st.title("📈 Model Performance Dashboard")
    st.write("Visual comparison of **Naive Bayes** and **SVM** models.")

    # ------------------------------
    # Accuracy Overview
    # ------------------------------
    st.subheader("🎯 Accuracy Overview")

    st.write(f"**Naive Bayes Accuracy:** {nb_acc:.4f}")
    st.progress(float(nb_acc))

    st.write(f"**SVM Accuracy:** {svm_acc:.4f}")
    st.progress(float(svm_acc))

    st.markdown("---")

    # ------------------------------
    # Accuracy Table & Bar Chart
    # ------------------------------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Accuracy Table")
        acc_df = pd.DataFrame({
            "Model": ["Naive Bayes", "SVM"],
            "Accuracy": [nb_acc, svm_acc]
        }).set_index("Model")
        st.dataframe(acc_df.style.format("{:.3f}"))

    with col2:
        st.subheader("Accuracy Bar Chart")
        fig, ax = plt.subplots()
        ax.bar(["Naive Bayes", "SVM"], [nb_acc, svm_acc])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Naive Bayes vs SVM")
        st.pyplot(fig)

    st.markdown("---")

    # ------------------------------
    # Confusion Matrices
    # ------------------------------
    st.subheader("📌 Confusion Matrices")

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.write("### Naive Bayes")
        fig_nb, ax_nb = plt.subplots()
        sns.heatmap(
            cm_nb, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"]
        )
        ax_nb.set_xlabel("Predicted")
        ax_nb.set_ylabel("Actual")
        st.pyplot(fig_nb)

    with col_cm2:
        st.write("### SVM")
        fig_svm, ax_svm = plt.subplots()
        sns.heatmap(
            cm_svm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"]
        )
        ax_svm.set_xlabel("Predicted")
        ax_svm.set_ylabel("Actual")
        st.pyplot(fig_svm)

    st.markdown("---")

    # ------------------------------
    # Detailed Metrics (Optional)
    # ------------------------------
    st.subheader("📋 Detailed Metrics Comparison")

    if nb_report and svm_report:
        metrics_df = pd.DataFrame([
            {
                "Model": "Naive Bayes",
                "Accuracy": nb_acc,
                "Precision": nb_report["weighted avg"]["precision"],
                "Recall": nb_report["weighted avg"]["recall"],
                "F1-Score": nb_report["weighted avg"]["f1-score"],
            },
            {
                "Model": "SVM",
                "Accuracy": svm_acc,
                "Precision": svm_report["weighted avg"]["precision"],
                "Recall": svm_report["weighted avg"]["recall"],
                "F1-Score": svm_report["weighted avg"]["f1-score"],
            }
        ]).set_index("Model")

        st.dataframe(metrics_df.style.format("{:.3f}"))
    else:
        st.info("Detailed classification metrics not available.")

# =====================================================
# SECTION 3: ABOUT PROJECT
# =====================================================
elif section == "ℹ️ About Project":
    st.title("ℹ️ Project Overview")

    st.markdown("""
    ## 🎯 Objective
    Build a system that classifies text as **Positive (1)** or **Negative (0)** using  
    **Naive Bayes** and **Support Vector Machine (SVM)**.

    ## 🧠 Approach
    - Merge training + testing dataset  
    - Convert text → numerical using **TF-IDF**  
    - Train Naive Bayes  
    - Train SVM  
    - Compare their performance  

    ## 📊 Evaluation Metrics Used
    - Accuracy  
    - Precision  
    - Recall  
    - F1-score  
    - Confusion Matrix  

    ## 🛠️ Technologies Used
    - Python  
    - Scikit-learn  
    - Pandas  
    - Matplotlib & Seaborn  
    - Streamlit  
    """)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:13px;'>"
    "📊 Sentiment Analysis Dashboard • Built with ❤️ using Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)
