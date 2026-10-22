
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Municipal Water Potability & Safety",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Municipal Drinking Water Potability & Safety Classification")
st.caption("AICTE IBM SkillsBuild Machine Learning & Applied AI Internship 2026")

st.markdown("""
### Project Overview
A supervised machine-learning project for classifying municipal drinking-water samples
as **potable (safe)** or **non-potable (unsafe)** using physicochemical water-quality
measurements.

**SDG 6 — Clean Water and Sanitation**
""")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Results", "🧪 Water Parameters", "🤖 ML Pipeline", "📌 Limitations & Future Scope"]
)

with tab1:
    st.subheader("Final Model Performance")
    cols = st.columns(5)
    metrics = [
        ("Accuracy", "66.62%"),
        ("Precision", "67.96%"),
        ("Potable Recall", "27.34%"),
        ("F1 Score", "39.00%"),
        ("ROC-AUC", "65.02%"),
    ]
    for col, (name, value) in zip(cols, metrics):
        col.metric(name, value)

    st.subheader("Safety-oriented evaluation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unsafe Recall", "91.75%")
    c2.metric("Unsafe → Safe Errors", "33")
    c3.metric("Test Samples", "656")
    c4.metric("Dataset Rows", "3,276")

    st.markdown("#### Confusion Matrix")
    cm = pd.DataFrame(
        [[367, 33], [186, 70]],
        index=["Actual Unsafe", "Actual Potable"],
        columns=["Predicted Unsafe", "Predicted Potable"]
    )
    st.dataframe(cm, use_container_width=True)

    st.info(
        "Interpretation: the final model is conservative and identifies most unsafe "
        "samples, but it also flags many potable samples as unsafe. It should be "
        "treated as a preliminary screening/risk-identification tool, not as a "
        "standalone water-safety certification system."
    )

with tab2:
    st.subheader("Input Water-Quality Parameters")
    features = [
        ("pH", "Acidity/alkalinity of water"),
        ("Hardness", "Mineral content, mainly calcium and magnesium"),
        ("Solids", "Total dissolved solids"),
        ("Chloramines", "Disinfectant-related measurement"),
        ("Sulfate", "Sulfate concentration"),
        ("Conductivity", "Electrical conductivity"),
        ("Organic Carbon", "Organic carbon concentration"),
        ("Trihalomethanes", "Disinfection by-product measurement"),
        ("Turbidity", "Cloudiness/particulate level"),
    ]
    for name, desc in features:
        st.markdown(f"**{name}** — {desc}")

with tab3:
    st.subheader("Machine-Learning Workflow")
    steps = [
        "1. Dataset inspection and missing-value analysis",
        "2. 80:20 stratified train/test split",
        "3. Missing-value handling using Iterative Imputation",
        "4. Feature standardization using StandardScaler",
        "5. Model comparison: KNN, Polynomial Logistic Regression and RBF-SVM",
        "6. 5-fold Stratified Cross-Validation",
        "7. Final selection: Iterative Imputation → StandardScaler → RBF-SVM",
        "8. Final held-out test evaluation using accuracy, precision, recall, F1 and ROC-AUC",
    ]
    for s in steps:
        st.markdown(s)

    st.subheader("Model comparison")
    comparison = pd.DataFrame({
        "Model": [
            "Median + KNN",
            "Median + Polynomial Logistic Regression",
            "Median + RBF-SVM",
            "Iterative + KNN",
            "Iterative + Polynomial Logistic Regression",
            "Iterative + RBF-SVM",
        ],
        "CV Accuracy": [66.18, 66.79, 67.98, 66.30, 66.83, 68.51],
        "Test Accuracy": [63.57, 65.70, 67.07, 62.96, 66.16, 66.62],
        "Test ROC-AUC": [61.63, 66.84, 64.87, 61.59, 67.22, 65.02],
    })
    st.dataframe(comparison, use_container_width=True)

with tab4:
    st.subheader("Limitations")
    for x in [
        "Overall predictive performance is moderate.",
        "Potable-class recall is low, so many potable samples are classified as unsafe.",
        "The dataset may not represent every municipal water system or geographic region.",
        "Physicochemical measurements alone do not capture every biological, chemical or operational water-safety factor.",
        "The model is not intended to replace laboratory testing or regulatory certification.",
    ]:
        st.markdown(f"• {x}")

    st.subheader("Future Scope")
    for x in [
        "Class weighting and decision-threshold optimization",
        "Ensemble and advanced machine-learning models",
        "Explainable AI for feature-level risk interpretation",
        "External validation on new municipal datasets",
        "Larger and more geographically diverse datasets",
        "Integration with real-time IoT water-quality sensors",
    ]:
        st.markdown(f"• {x}")

st.divider()
st.markdown(
    "**Developed by Harshitha S.**  \n"
    "B.Tech Computer Engineering | AICTE IBM SkillsBuild Internship 2026"
)
