import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dijital Bağımlılık • Bilimsel Analiz Paneli", layout="wide")

st.title("📱 Dijital Bağımlılık • Uyku • Sağlık • Sosyal Yaşam (Bilimsel Dashboard)")
st.caption("İstatistiksel testler + etki büyüklüğü + çoklu regresyon + risk segmentasyonu (phone_addiction_dataset.csv)")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

uploaded = st.sidebar.file_uploader("CSV yükle (opsiyonel)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = load_data("phone_addiction_dataset.csv")
    except Exception:
        st.error("CSV bulunamadı. phone_addiction_dataset.csv dosyasını app.py ile aynı klasöre koy veya soldan yükle.")
        st.stop()

# Kolon adı düzeltme (bazı datasetlerde typo olabiliyor)
if "Interllectual_Performance" not in df.columns and "Intellectual_Performance" in df.columns:
    df.rename(columns={"Intellectual_Performance": "Interllectual_Performance"}, inplace=True)

# Sayısal dönüşüm
possible_numeric = [
    "Age", "Daily_Usage_Hours", "Weekend_Usage_Hours", "Sleep_Hours",
    "Interllectual_Performance", "Social_Interactions", "Exercise_Hours",
    "Anxiety_Level", "Depression_Level", "Self_Esteem", "Screen_Time_Before_Bed",
    "Phone_Checks_Per_Day", "Apps_Used_Daily", "Time_on_Social_Media",
    "Time_on_Gaming", "Time_on_Education", "Family_Communication",
]
for c in possible_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Temel filtrelenebilir kolonlar
base_cols = [c for c in ["Age", "Gender", "Addiction_Level"] if c in df.columns]
df = df.dropna(subset=base_cols)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filtreler")

if "Gender" in df.columns:
    genders = sorted(df["Gender"].dropna().unique())
    sel_gender = st.sidebar.multiselect("Cinsiyet", genders, default=genders)
else:
    sel_gender = None

if "Addiction_Level" in df.columns:
    levels = sorted(df["Addiction_Level"].dropna().unique())
    sel_level = st.sidebar.multiselect("Bağımlılık Seviyesi", levels, default=levels)
else:
    sel_level = None

if "Age" in df.columns:
    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Yaş Aralığı", min_age, max_age, (min_age, max_age))
else:
    age_range = None

filtered = df.copy()
if sel_gender is not None:
    filtered = filtered[filtered["Gender"].isin(sel_gender)]
if sel_level is not None:
    filtered = filtered[filtered["Addiction_Level"].isin(sel_level)]
if age_range is not None:
    filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

# -----------------------------
# KPI
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Kayıt Sayısı", f"{len(filtered):,}")
if "Daily_Usage_Hours" in filtered.columns:
    k2.metric("Ort. Günlük Kullanım (saat)", f"{filtered['Daily_Usage_Hours'].mean():.2f}")
if "Sleep_Hours" in filtered.columns:
    k3.metric("Ort. Uyku (saat)", f"{filtered['Sleep_Hours'].mean():.2f}")
if "Time_on_Social_Media" in filtered.columns:
    k4.metric("Ort. Sosyal Medya (saat)", f"{filtered['Time_on_Social_Media'].mean():.2f}")

st.divider()

# -----------------------------
# Helpers: Effect sizes
# -----------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp

def eta_squared_from_anova(anova_table: pd.DataFrame, effect_row_name: str) -> float:
    # eta^2 = SS_effect / SS_total
    try:
        ss_effect = anova_table.loc[effect_row_name, "sum_sq"]
        ss_total = anova_table["sum_sq"].sum()
        return float(ss_effect / ss_total) if ss_total != 0 else np.nan
    except Exception:
        return np.nan

# -----------------------------
# Tabs
# -----------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧾 Veri",
    "📊 Keşif",
    "🧪 ANOVA / t-test",
    "📏 Etki Büyüklüğü",
    "📈 Çoklu Regresyon",
    "🧩 Risk Segmentasyonu",
    "📌 Rapor Özeti",
])

# -----------------------------
# TAB 0: Data overview
# -----------------------------
with tab0:
    st.subheader("Kolonlar ve Eksikler")
    info = pd.DataFrame({
        "column": filtered.columns,
        "dtype": [str(filtered[c].dtype) for c in filtered.columns],
        "missing_count": [int(filtered[c].isna().sum()) for c in filtered.columns],
        "unique_count": [int(filtered[c].nunique(dropna=True)) for c in filtered.columns],
    })
    st.dataframe(info, use_container_width=True)
    st.subheader("Veri önizleme")
    st.dataframe(filtered.head(200), use_container_width=True)

# -----------------------------
# TAB 1: EDA
# -----------------------------
with tab1:
    st.subheader("Dağılımlar ve Kıyaslar")
    numeric_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    cat_cols = [c for c in filtered.columns if filtered[c].dtype == "object"]

    c1, c2 = st.columns(2)
    with c1:
        if numeric_cols:
            col = st.selectbox("Histogram için sayısal kolon", numeric_cols, index=0)
            fig = px.histogram(filtered, x=col, nbins=30, title=f"{col} dağılımı")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if cat_cols:
            cat = st.selectbox("Bar için kategorik kolon", cat_cols, index=0)
            vc = filtered[cat].value_counts().reset_index()
            vc.columns = [cat, "count"]
            fig = px.bar(vc, x=cat, y="count", title=f"{cat} frekans")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Boxplot (grup → metrik)")
    group_candidates = [c for c in ["Addiction_Level", "Gender", "Phone_Usage_Purpose", "Location"] if c in filtered.columns]
    metric_candidates = [c for c in numeric_cols if c not in ["Age"]]

    if group_candidates and metric_candidates:
        g = st.selectbox("Gruplama", group_candidates, index=0)
        m = st.selectbox("Metrik", metric_candidates, index=metric_candidates.index("Sleep_Hours") if "Sleep_Hours" in metric_candidates else 0)
        fig = px.box(filtered, x=g, y=m, points="outliers", title=f"{g} → {m}")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2: ANOVA / t-test
# -----------------------------
with tab2:
    st.subheader("ANOVA / t-test (istatistiksel anlamlılık)")

    group_col = st.selectbox("Gruplama değişkeni (kategori)", [c for c in ["Addiction_Level", "Gender", "Phone_Usage_Purpose", "Location"] if c in filtered.columns], index=0)
    y_col = st.selectbox("Sonuç metriği (sayısal)", [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])], index=0)

    data = filtered[[group_col, y_col]].dropna()
    groups = [g[y_col].values for _, g in data.groupby(group_col)]

    if data[group_col].nunique() == 2:
        st.markdown("### t-test (2 grup)")
        gnames = list(data[group_col].unique())
        a = data[data[group_col] == gnames[0]][y_col].values
        b = data[data[group_col] == gnames[1]][y_col].values

        # Welch t-test (varyans eşit varsaymıyoruz)
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        d = cohens_d(a, b)

        st.write(f"Gruplar: **{gnames[0]}** vs **{gnames[1]}**")
        st.write(f"t-istatistiği: **{t_stat:.3f}**")
        st.write(f"p-değeri: **{p_val:.6f}**")
        st.write(f"Cohen's d (etki): **{d:.3f}**  (|d|: 0.2 küçük, 0.5 orta, 0.8 büyük ~ kural-of-thumb)")

    elif data[group_col].nunique() >= 3:
        st.markdown("### ANOVA (3+ grup)")
        f_stat, p_val = stats.f_oneway(*groups)
        st.write(f"F-istatistiği: **{f_stat:.3f}**")
        st.write(f"p-değeri: **{p_val:.6f}**")

        st.markdown("### ANOVA tablosu (OLS)")
        # OLS ANOVA via statsmodels
        # Güvenli formula için kolon adlarını backtick değil; basit isimler varsayıyoruz.
        # Eğer boşluklu kolon olursa burada patlar; bu datasette yok.
        model = smf.ols(f"{y_col} ~ C({group_col})", data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.dataframe(anova_table)

    else:
        st.warning("Seçtiğin gruplama sütununda yeterli grup yok.")

# -----------------------------
# TAB 3: Effect sizes (more complete)
# -----------------------------
with tab3:
    st.subheader("Etki büyüklüğü (effect size) – sadece p değil, etkiyi de ölç")

    group_col = st.selectbox("Etki için gruplama (kategori)", [c for c in ["Addiction_Level", "Gender", "Phone_Usage_Purpose", "Location"] if c in filtered.columns], index=0, key="es_group")
    y_col = st.selectbox("Etki metriği (sayısal)", [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])], index=0, key="es_y")

    data = filtered[[group_col, y_col]].dropna()

    if data[group_col].nunique() == 2:
        gnames = list(data[group_col].unique())
        a = data[data[group_col] == gnames[0]][y_col].values
        b = data[data[group_col] == gnames[1]][y_col].values
        d = cohens_d(a, b)

        st.write(f"**Cohen's d** ({gnames[0]} vs {gnames[1]}): **{d:.3f}**")
        st.caption("Kaba yorum: |d|≈0.2 küçük, 0.5 orta, 0.8 büyük (bağlama göre değişir).")

    elif data[group_col].nunique() >= 3:
        model = smf.ols(f"{y_col} ~ C({group_col})", data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        # statsmodels anova tablosunda efekt satırı genelde C(group_col)
        effect_name = f"C({group_col})"
        eta2 = eta_squared_from_anova(anova_table, effect_name)

        st.dataframe(anova_table)
        st.write(f"**Eta-squared (η²)**: **{eta2:.3f}**")
        st.caption("Kaba yorum: η²≈0.01 küçük, 0.06 orta, 0.14 büyük (kural-of-thumb).")
    else:
        st.warning("Etki hesabı için yeterli grup yok.")

# -----------------------------
# TAB 4: Multiple regression
# -----------------------------
with tab4:
    st.subheader("Çoklu Regresyon (çok değişkenli etki analizi)")
    st.caption("Amaç: Bir sonucu (örn. Sleep_Hours) birden çok faktörle aynı anda açıklamak.")

    # Target (Y)
    numeric_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    default_y = "Sleep_Hours" if "Sleep_Hours" in numeric_cols else numeric_cols[0]
    y = st.selectbox("Hedef değişken (Y)", numeric_cols, index=numeric_cols.index(default_y))

    # Candidate predictors (X)
    # Hem sayısal hem bazı kategorikler
    candidate_numeric = [c for c in numeric_cols if c != y and c not in ["Age"]]
    candidate_cat = [c for c in ["Addiction_Level", "Gender", "Phone_Usage_Purpose", "Location"] if c in filtered.columns]

    st.markdown("### Bağımsız değişkenleri seç (X)")
    x_num = st.multiselect("Sayısal X'ler", candidate_numeric, default=[c for c in ["Daily_Usage_Hours", "Time_on_Social_Media", "Screen_Time_Before_Bed", "Phone_Checks_Per_Day"] if c in candidate_numeric])
    x_cat = st.multiselect("Kategorik X'ler (otomatik dummy)", candidate_cat, default=[c for c in ["Addiction_Level", "Gender"] if c in candidate_cat])

    if len(x_num) + len(x_cat) < 1:
        st.warning("En az 1 bağımsız değişken seçmelisin.")
    else:
        # Prepare data
        cols_needed = [y] + x_num + x_cat
        data = filtered[cols_needed].dropna().copy()

        # Build formula: y ~ x1 + x2 + C(cat1) + C(cat2)
        formula_parts = []
        formula_parts += x_num
        formula_parts += [f"C({c})" for c in x_cat]
        formula = f"{y} ~ " + " + ".join(formula_parts)

        st.code(formula)

        model = smf.ols(formula, data=data).fit()
        st.markdown("### Model Özeti (kısa)")
        st.write(f"R²: **{model.rsquared:.3f}** | Adj. R²: **{model.rsquared_adj:.3f}** | n: **{int(model.nobs)}**")

        # Coeff table
        coef = pd.DataFrame({
            "term": model.params.index,
            "coef": model.params.values,
            "p_value": model.pvalues.values,
            "std_err": model.bse.values,
        }).sort_values("p_value")

        st.dataframe(coef, use_container_width=True)

        st.markdown("### Tahmin vs Gerçek (hızlı kontrol)")
        data["_pred"] = model.predict(data)
        fig = px.scatter(data, x="_pred", y=y, trendline="ols", title="Tahmin edilen vs Gerçek")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Not: Regresyon ilişkiyi gösterir; nedensellik için deneysel/yarı-deneysel tasarım gerekir.")

# -----------------------------
# TAB 5: Risk segmentation (clustering)
# -----------------------------
with tab5:
    st.subheader("Risk Segmentasyonu (kümeleme)")
    st.caption("Amaç: Benzer risk profillerini otomatik gruplamak (örn. yüksek kullanım + düşük uyku + yüksek anksiyete).")

    # Segmentation features (numerics)
    seg_features_default = [c for c in [
        "Daily_Usage_Hours", "Time_on_Social_Media", "Screen_Time_Before_Bed",
        "Phone_Checks_Per_Day", "Sleep_Hours", "Anxiety_Level", "Depression_Level",
        "Exercise_Hours", "Social_Interactions", "Family_Communication", "Self_Esteem"
    ] if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c])]

    seg_features = st.multiselect("Segmentasyon değişkenleri", 
                                 [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])],
                                 default=seg_features_default)

    k = st.slider("Küme sayısı (K)", 2, 8, 4)

    if len(seg_features) < 2:
        st.warning("Segmentasyon için en az 2 sayısal değişken seç.")
    else:
        data = filtered[seg_features].dropna().copy()
        if len(data) < 50:
            st.warning("Kümeleme için veri çok az (filtreleri genişlet).")
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(data.values)

            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)

            data_seg = data.copy()
            data_seg["Segment"] = labels

            st.markdown("### Segment profilleri (ortalama)")
            prof = data_seg.groupby("Segment")[seg_features].mean().round(2)
            st.dataframe(prof, use_container_width=True)

            # Görselleştirme: 2 boyuta indir (PCA yoksa basit çift eksen seçimi)
            st.markdown("### Segment görselleştirme (2 değişken seç)")
            x = st.selectbox("X", seg_features, index=0)
            y = st.selectbox("Y", seg_features, index=1)
            plot_df = data_seg[[x, y, "Segment"]].copy()

            fig = px.scatter(plot_df, x=x, y=y, color="Segment", title="Segmentler (seçilen iki eksende)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Risk etiketi (kural tabanlı)")
            # Basit risk skoru: yüksek kullanım + düşük uyku + yüksek anksiyete/depresyon
            # (Eşikler dataset dağılımından otomatik)
            risk_cols = [c for c in ["Daily_Usage_Hours", "Time_on_Social_Media", "Sleep_Hours", "Anxiety_Level", "Depression_Level"] if c in data_seg.columns]
            if len(risk_cols) >= 3:
                tmp = data_seg[risk_cols].copy()
                # otomatik eşik: üst %75 kullanım metrikleri, alt %25 uyku
                high_use = []
                if "Daily_Usage_Hours" in tmp.columns: high_use.append(("Daily_Usage_Hours", tmp["Daily_Usage_Hours"].quantile(0.75), "high"))
                if "Time_on_Social_Media" in tmp.columns: high_use.append(("Time_on_Social_Media", tmp["Time_on_Social_Media"].quantile(0.75), "high"))
                low_sleep_thr = tmp["Sleep_Hours"].quantile(0.25) if "Sleep_Hours" in tmp.columns else None

                def risk_flag(row):
                    score = 0
                    for col, thr, kind in high_use:
                        if row[col] >= thr: score += 1
                    if low_sleep_thr is not None and "Sleep_Hours" in row.index:
                        if row["Sleep_Hours"] <= low_sleep_thr: score += 1
                    if "Anxiety_Level" in row.index:
                        if row["Anxiety_Level"] >= tmp["Anxiety_Level"].quantile(0.75): score += 1
                    if "Depression_Level" in row.index:
                        if row["Depression_Level"] >= tmp["Depression_Level"].quantile(0.75): score += 1
                    return score

                data_seg["_risk_score"] = data_seg.apply(risk_flag, axis=1)
                # Segment başına ortalama risk
                risk_by_seg = data_seg.groupby("Segment")["_risk_score"].mean().sort_values(ascending=False).reset_index()
                risk_by_seg.rename(columns={"_risk_score": "avg_risk_score"}, inplace=True)

                fig = px.bar(risk_by_seg, x="Segment", y="avg_risk_score", title="Segment başına ortalama risk skoru")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Risk skoru tamamen kural tabanlıdır; politika/uzman görüşüyle kalibre edilmelidir.")

# -----------------------------
# TAB 6: Report summary
# -----------------------------
with tab6:
    st.subheader("Rapor Özeti (tek sayfa)")
    st.write("Aşağıdaki blok, seçtiğin filtrelerle oluşan veriden otomatik özet çıkarır.")

    # hızlı özet metrikleri
    lines = []
    lines.append(f"- Kayıt sayısı: **{len(filtered):,}**")

    if "Addiction_Level" in filtered.columns and "Sleep_Hours" in filtered.columns:
        grp = filtered.groupby("Addiction_Level")["Sleep_Hours"].mean().sort_values(ascending=False)
        if len(grp) >= 2:
            best = grp.index[0]
            worst = grp.index[-1]
            lines.append(f"- Uyku (ortalama): en yüksek **{best}**, en düşük **{worst}**.")

    if "Daily_Usage_Hours" in filtered.columns and "Sleep_Hours" in filtered.columns:
        corr = filtered[["Daily_Usage_Hours", "Sleep_Hours"]].dropna().corr().iloc[0, 1]
        lines.append(f"- Günlük kullanım ↔ uyku korelasyonu: **{corr:.3f}** (yaklaşık).")

    if "Time_on_Social_Media" in filtered.columns and "Depression_Level" in filtered.columns:
        corr = filtered[["Time_on_Social_Media", "Depression_Level"]].dropna().corr().iloc[0, 1]
        lines.append(f"- Sosyal medya ↔ depresyon korelasyonu: **{corr:.3f}** (yaklaşık).")

    st.markdown("\n".join(lines))

    st.markdown("### Hızlı yorum şablonu")
    st.code(
        "Bu çalışma, dijital kullanım göstergeleri (günlük kullanım, sosyal medya süresi, yatmadan önce ekran süresi vb.)\n"
        "ile uyku ve psikolojik sağlık göstergeleri (anksiyete, depresyon, özsaygı, sosyal etkileşim) arasındaki ilişkileri\n"
        "tanımlayıcı istatistikler, grup karşılaştırmaları (ANOVA/t-test), etki büyüklüğü ölçütleri ve çoklu regresyon modeli ile\n"
        "incelemiştir. Ayrıca risk segmentasyonu ile benzer risk profilleri kümelenmiştir.",
        language="text"
    )

st.divider()
st.subheader("Veri önizleme")
st.dataframe(filtered.head(200), use_container_width=True)
