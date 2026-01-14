import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Dijital Bağımlılık Analizi", layout="wide")

st.title("📱 Dijital Bağımlılık • Uyku • Sağlık • Sosyal Yaşam Dashboard")
st.caption("phone_addiction_dataset.csv üzerinden etki analizi ve görselleştirme")

# -----------------------------
# Data Load
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

# -----------------------------
# Column Standardization / Safety
# -----------------------------
# Bazı datasetlerde kolon isimlerinde yazım farkları olabilir.
# Bu dataset özelinde: "Interllectual_Performance" (iki L) şeklinde geliyor.
# Eğer farklı gelirse yakalayalım.
if "Interllectual_Performance" not in df.columns and "Intellectual_Performance" in df.columns:
    df.rename(columns={"Intellectual_Performance": "Interllectual_Performance"}, inplace=True)

# Numerik dönüşüm
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

# Eksik temel alanları temizle
base_cols = [c for c in ["Age", "Gender", "Addiction_Level"] if c in df.columns]
df = df.dropna(subset=base_cols)

# -----------------------------
# Sidebar Filters
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
# KPIs
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
# Tabs
# -----------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🧾 Veri & Kolonlar",
    "📊 Dağılımlar",
    "⚖️ Kıyas (Grup Karşılaştırma)",
    "📈 İlişki (Scatter)",
    "🔗 Korelasyon"
])

# -----------------------------
# TAB 0: Data overview
# -----------------------------
with tab0:
    st.subheader("Kolonlar (Ana Başlıklar) ve Veri Tipleri")
    info = pd.DataFrame({
        "column": filtered.columns,
        "dtype": [str(filtered[c].dtype) for c in filtered.columns],
        "missing_count": [int(filtered[c].isna().sum()) for c in filtered.columns],
        "unique_count": [int(filtered[c].nunique(dropna=True)) for c in filtered.columns],
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("Veri Önizleme")
    st.dataframe(filtered.head(200), use_container_width=True)

# -----------------------------
# TAB 1: Distributions
# -----------------------------
with tab1:
    st.subheader("Seçili Sayısal Kolon Dağılımı (Histogram)")
    numeric_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    if numeric_cols:
        col = st.selectbox("Sayısal kolon seç", numeric_cols, index=min(0, len(numeric_cols)-1))
        fig = px.histogram(filtered, x=col, nbins=30, title=f"{col} Dağılımı")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sayısal kolon bulunamadı.")

    st.subheader("Kategori Dağılımı (Bar)")
    cat_cols = [c for c in filtered.columns if filtered[c].dtype == "object"]
    if cat_cols:
        cat = st.selectbox("Kategorik kolon seç", cat_cols, index=0)
        vc = filtered[cat].value_counts().reset_index()
        vc.columns = [cat, "count"]
        fig = px.bar(vc, x=cat, y="count", title=f"{cat} Frekans")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2: Group comparisons
# -----------------------------
with tab2:
    st.subheader("Grup Karşılaştırma (Addiction_Level / Gender / Purpose vs metrik)")

    # Grup kolonları
    group_candidates = [c for c in ["Addiction_Level", "Gender", "Phone_Usage_Purpose", "Location"] if c in filtered.columns]
    metric_candidates = [c for c in [
        "Sleep_Hours", "Interllectual_Performance", "Social_Interactions", "Exercise_Hours",
        "Anxiety_Level", "Depression_Level", "Self_Esteem", "Family_Communication",
        "Daily_Usage_Hours", "Weekend_Usage_Hours", "Phone_Checks_Per_Day",
        "Screen_Time_Before_Bed", "Time_on_Social_Media", "Time_on_Gaming", "Time_on_Education"
    ] if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c])]

    if group_candidates and metric_candidates:
        group_col = st.selectbox("Gruplama (X)", group_candidates, index=0)
        metric_col = st.selectbox("Metrik (Y)", metric_candidates, index=0)

        st.markdown("**Boxplot (dağılım + medyan):**")
        fig = px.box(filtered, x=group_col, y=metric_col, points="outliers",
                     title=f"{group_col} → {metric_col} (Boxplot)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Ortalama karşılaştırma (bar):**")
        grp = filtered.groupby(group_col)[metric_col].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(grp, x=group_col, y=metric_col, title=f"{group_col} → {metric_col} (Ortalama)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Özet tablo:**")
        summary = filtered.groupby(group_col)[metric_col].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Grup karşılaştırma için uygun kolonlar bulunamadı (kategorik + sayısal gerekli).")

# -----------------------------
# TAB 3: Scatter relationships
# -----------------------------
with tab3:
    st.subheader("İki değişken ilişkisi (Scatter + trend)")

    x_candidates = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    y_candidates = x_candidates.copy()

    if x_candidates and y_candidates:
        x = st.selectbox("X (neden/etken)", x_candidates, index=x_candidates.index("Daily_Usage_Hours") if "Daily_Usage_Hours" in x_candidates else 0)
        y = st.selectbox("Y (sonuç/etki)", y_candidates, index=y_candidates.index("Sleep_Hours") if "Sleep_Hours" in y_candidates else 0)

        color_by = st.selectbox("Renklendir (kategori)", ["(yok)"] + [c for c in ["Addiction_Level", "Gender", "Phone_Usage_Purpose"] if c in filtered.columns], index=0)

        if color_by == "(yok)":
            fig = px.scatter(filtered, x=x, y=y, trendline="ols", title=f"{x} ↔ {y}")
        else:
            fig = px.scatter(filtered, x=x, y=y, color=color_by, trendline="ols", title=f"{x} ↔ {y} (Color: {color_by})")

        st.plotly_chart(fig, use_container_width=True)

        # Basit korelasyon değeri
        corr = filtered[[x, y]].dropna().corr().iloc[0, 1]
        st.info(f"Pearson korelasyon (yaklaşık): **{corr:.3f}**  (nedensellik kanıtı değildir)")
    else:
        st.info("Scatter için yeterli sayısal kolon yok.")

# -----------------------------
# TAB 4: Correlation heatmap
# -----------------------------
with tab4:
    st.subheader("Korelasyon Isı Haritası (Sayısal kolonlar)")
    numeric = filtered.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        # Çok kolon varsa ana kolonlara indir
        main = [c for c in [
            "Daily_Usage_Hours", "Weekend_Usage_Hours", "Sleep_Hours",
            "Interllectual_Performance", "Anxiety_Level", "Depression_Level",
            "Self_Esteem", "Social_Interactions", "Exercise_Hours",
            "Phone_Checks_Per_Day", "Screen_Time_Before_Bed", "Apps_Used_Daily",
            "Time_on_Social_Media", "Time_on_Gaming", "Time_on_Education",
            "Family_Communication"
        ] if c in numeric.columns]

        use_cols = main if len(main) >= 2 else numeric.columns.tolist()
        corr = numeric[use_cols].corr()

        fig = px.imshow(corr, text_auto=".2f", title="Korelasyon Matrisi", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("İpucu: |corr| yüksekse ilişki güçlü olabilir; ama bu tek başına 'sebep' demek değildir.")
    else:
        st.info("Korelasyon için en az 2 sayısal kolon gerekir.")
