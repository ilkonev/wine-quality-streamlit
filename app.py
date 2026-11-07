import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")

@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / "test.csv")

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

model = load_model()
config = json.loads((APP_DIR / "config.json").read_text())
test_df = load_test()

features = list(model.feature_names_in_)
X_test = test_df[features]
y_test = test_df["target"].to_numpy()
proba_test = test_proba(model, X_test)

threshold = st.sidebar.slider("Порог", 0.0, 1.0, float(config["threshold"]), 0.001, key="threshold")

tab1, tab2, tab3, tab4 = st.tabs(["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Коэффициенты модели"])

with tab1:
    values = {}
    for f in features:
        if f == "is_red":
            continue
        lo = float(X_test[f].min())
        hi = float(X_test[f].max())
        med = float(X_test[f].median())
        values[f] = st.number_input(f, min_value=lo, max_value=hi, value=med)
    
    colour = st.radio("Цвет", ["красное", "белое"], horizontal=True)
    values["is_red"] = 1.0 if colour == "красное" else 0.0

    p = float(model.predict_proba(pd.DataFrame([values])[features])[0, 1])
    st.metric("Вероятность, что вино хорошее", f"{p:.3f}")
    if p >= threshold:
        st.success("Решение: хорошее вино")
    else:
        st.warning("Решение: не хорошее вино")

with tab2:
    pred = (proba_test >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{precision:.3f}")
    c2.metric("Recall", f"{recall:.3f}")
    c3.metric("F1", f"{f1:.3f}")

    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=["не хорошее", "хорошее"]).plot(ax=ax, colorbar=False )
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr)
    fpr_point = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ax.scatter([fpr_point], [recall])
    
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC-кривая")
    st.pyplot(fig)
    plt.close(fig)

    pr, rc, _ = metrics.precision_recall_curve(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rc, pr)
    ax.scatter([recall], [precision])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR-кривая")
    st.pyplot(fig)
    plt.close(fig)

    st.info("Если порог увеличить, подборка станет меньше, но в ней будет больше действительно хороших вин. " \
    "\nЕсли порог уменьшить, подборка расширится, но в неё чаще будут попадать посредственные вина")

with tab3:
    up = st.file_uploader("CSV-файл с признаками вин", type=["csv"])
    if up is not None:
        batch = pd.read_csv(up)
        missing = [f for f in features if f not in batch.columns]
        if missing:
            st.error("В файле не хватает столбцов: " + ", ".join(missing))
        else:
            probs = model.predict_proba(batch[features])[:, 1]
            out = batch.copy()
            out["proba"] = probs
            out["decision"] = (probs >= st.session_state["threshold"]).astype(int)
            st.dataframe(out)
            st.download_button(
                "Скачать результат",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

with tab4:
    coefs = pd.Series(model.named_steps["logisticregression"].coef_[0], index=features).sort_values()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(coefs.index, coefs.values)
    ax.axvline(0, lw=0.8)
    ax.set_xlabel("Коэффициент")
    st.pyplot(fig)
    plt.close(fig)
    
    st.write("За хорошее вино сильнее всего: " + ", ".join(coefs.sort_values(ascending=False).index[:3]))
    st.write("Против хорошего вина сильнее всего: " + ", ".join(coefs.index[:3]))