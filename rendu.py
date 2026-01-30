import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"

adult_clean = DATA_PATH / "adult_clean.csv"
train_fair = DATA_PATH / "train_fair.csv"
rfc_naive = DATA_PATH / "rfc_naive.pkl"
rfc_resampled = DATA_PATH / "rfc_resampled.pkl"
rfc_fair = DATA_PATH / "rfc_fair.pkl"
scaler_naive = DATA_PATH / "scaler_naive.pkl"
scaler_resampled = DATA_PATH / "scaler_resampled.pkl"
scaler_fair = DATA_PATH / "scaler_fair.pkl"

# Cache
@st.cache_data
def load_data():
    return pd.read_csv(adult_clean)

@st.cache_resource
def load_models():
    return {
        "naive": joblib.load(rfc_naive),
        "resampled": joblib.load(rfc_resampled),
        "fair": joblib.load(rfc_fair),
    }

df = load_data()
models = load_models()

st.sidebar.title("🎯 Modèles")
model_choice = st.sidebar.selectbox("Choisir", ["fair", "resampled", "naive"])

page = st.sidebar.selectbox(
    "Pages",
    ["🏠 Accueil", "📊 Exploration", "⚠️ Biais", "🤖 Modélisation", "🧪 Test modèle"]
)

if page == "🏠 Accueil":
    st.title("🧑‍💼 Détection Biais : Adult Income")
    st.markdown("**Problème** : Prédire >50K$/an. **Biais** : Genre Female défavorisée.")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Lignes", f"{len(df):,}")
    with col2: st.metric("Colonnes", df.shape[1])
    with col3: st.metric("Manquants %", f"{df.isnull().sum().sum()/len(df)*100:.1f}%")
    with col4: st.metric(">50K %", f"{df['income'].mean():.1%}")

    st.dataframe(df.head(), use_container_width=True)
    st.caption("gender:0=F/1=M")

elif page == "📊 Exploration":
    st.header("Exploration")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Âge médian", df["age"].median())
    with col2: st.metric("Âge moy", f"{df['age'].mean():.0f}")
    with col3: st.metric("Male %", f"{df['gender'].mean():.0%}")
    with col4: st.metric("Income 1 %", f"{df['income'].mean():.0%}")

    fig1 = px.histogram(df, x="income", title="Distribution cible")
    st.plotly_chart(fig1, use_container_width=True)

    group = df.groupby(["gender", "income"]).size().reset_index(name="count")
    fig2 = px.bar(group, x="gender", y="count", color="income", title=">50K par Genre")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, x="gender", y="age", color="income", title="Âge par Genre")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "⚠️ Biais":
    st.header("Biais Genre")

    scaler_fair_loaded = joblib.load(scaler_fair)

    # Colonnes attendues par le scaler (noms + ordre)
    expected_cols = list(scaler_fair_loaded.feature_names_in_)  # peut inclure income [web:76]

    df_all = df.copy()
    for c in expected_cols:
        if c not in df_all.columns:
            df_all[c] = 0

    X_all = df_all[expected_cols]

    X_all_scaled = scaler_fair_loaded.transform(X_all)
    X_all_scaled = pd.DataFrame(X_all_scaled, columns=expected_cols)

    model_cols = [c for c in expected_cols if c != "income"]
    X_demo_scaled = X_all_scaled[model_cols]

    y_pred_demo = models["fair"].predict(X_demo_scaled)

    p_f = np.sum((df["gender"] == 0) & (y_pred_demo == 1)) / np.sum(df["gender"] == 0)
    p_m = np.sum((df["gender"] == 1) & (y_pred_demo == 1)) / np.sum(df["gender"] == 1)

    st.metric("Parité Démographique", f"{abs(p_f - p_m):.3f}")
    st.metric("Disparate Impact", f"{min(p_f / p_m, p_m / p_f):.3f}")

    biais_df = pd.DataFrame({"Genre": ["Female", "Male"], "Prédit >50K": [p_f, p_m]})
    fig = px.bar(biais_df, x="Genre", y="Prédit >50K", title="Taux >50K prédit par genre")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Interprétation : si DI proche de 1 et DP proche de 0 → biais réduit.")

elif page == "🤖 Modélisation":
    st.header("Modèles Comparaison")

    test_df = pd.read_csv(train_fair)

    scalers = {
        "naive": joblib.load(scaler_naive),
        "resampled": joblib.load(scaler_resampled),
        "fair": joblib.load(scaler_fair),
    }

    results = {}

    for name, model in models.items():
        scaler = scalers[name]
        expected_cols = list(scaler.feature_names_in_)  # [web:76]

        df_all = test_df.copy()
        for c in expected_cols:
            if c not in df_all.columns:
                df_all[c] = 0

        X_all = df_all[expected_cols]
        y_test = df_all["income"]

        X_all_scaled = scaler.transform(X_all)
        X_all_scaled = pd.DataFrame(X_all_scaled, columns=expected_cols)

        model_cols = [c for c in expected_cols if c != "income"]
        X_test_scaled = X_all_scaled[model_cols]

        y_pred = model.predict(X_test_scaled)
        results[name] = accuracy_score(y_test, y_pred)

    df_res = pd.DataFrame([results]).T.reset_index()
    df_res.columns = ["Modèle", "Accuracy"]
    st.dataframe(df_res.style.highlight_max(), use_container_width=True)

    fig = px.bar(df_res, x="Modèle", y="Accuracy", title="Accuracy par modèle")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🧪 Test modèle":
    st.header("🧪 Tester un modèle (entrée manuelle)")

    chosen_model = st.selectbox("Modèle", ["fair", "resampled", "naive"])
    model = models[chosen_model]

    scalers = {
        "naive": joblib.load(scaler_naive),
        "resampled": joblib.load(scaler_resampled),
        "fair": joblib.load(scaler_fair),
    }
    scaler = scalers[chosen_model]

    expected_cols = list(scaler.feature_names_in_)  # [web:76]

    # Form = toutes les valeurs sont envoyées d'un coup au submit [web:126][web:125]
    with st.form("manual_input_form"):
        age = st.number_input("age", min_value=0, max_value=100, value=30, step=1)
        fnlwgt = st.number_input("fnlwgt", min_value=0, value=100000, step=1000)
        edu_num = st.number_input("educational-num", min_value=0, max_value=20, value=10, step=1)
        race = st.selectbox("race (0=Other, 1=White)", [0, 1], index=1)
        gender = st.selectbox("gender (0=Female, 1=Male)", [0, 1], index=1)
        cap_gain = st.number_input("capital-gain", min_value=0, value=0, step=100)
        cap_loss = st.number_input("capital-loss", min_value=0, value=0, step=100)
        hpw = st.number_input("hours-per-week", min_value=1, max_value=99, value=40, step=1)
        nat = st.selectbox("native-country (0=US, 1=ExPat)", [0, 1], index=0)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        # 1) Ligne complète attendue par scaler
        row = {c: 0 for c in expected_cols}

        # 2) Features de base
        if "age" in row: row["age"] = age
        if "fnlwgt" in row: row["fnlwgt"] = fnlwgt
        if "educational-num" in row: row["educational-num"] = edu_num
        if "race" in row: row["race"] = race
        if "gender" in row: row["gender"] = gender
        if "capital-gain" in row: row["capital-gain"] = cap_gain
        if "capital-loss" in row: row["capital-loss"] = cap_loss
        if "hours-per-week" in row: row["hours-per-week"] = hpw
        if "native-country" in row: row["native-country"] = nat

        # 3) Si scaler attend income, on met dummy 0
        if "income" in row:
            row["income"] = 0

        X_one = pd.DataFrame([row], columns=expected_cols)

        X_scaled = scaler.transform(X_one)
        X_scaled = pd.DataFrame(X_scaled, columns=expected_cols)

        model_cols = [c for c in expected_cols if c != "income"]
        X_model = X_scaled[model_cols]

        pred = int(model.predict(X_model)[0])

        st.write("Entrée envoyée au modèle (aperçu) :")
        st.dataframe(X_one.head(1), use_container_width=True)

        if pred == 1:
            st.success("Résultat : prédiction > 50K")
        else:
            st.warning("Résultat : prédiction ≤ 50K")

st.sidebar.markdown("UCI Adult | Fairness via Massaging")
