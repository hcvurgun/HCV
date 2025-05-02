import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

st.set_page_config(page_title="İstatistiksel Analiz Uygulaması", layout="wide")
st.title("Web Tabanlı İstatistiksel Analiz Aracı")

uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Veri Önizlemesi")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        st.subheader("Temel İstatistiksel Özellikler")
        st.write(df[numeric_cols].describe())

        st.subheader("Grafikler")
        col = st.selectbox("Bir sütun seçin", numeric_cols)

        tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "Kutu Grafiği", "Dağılım Grafiği", "X̄-R Kontrol Grafiği"])

        with tab1:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

        with tab3:
            other_col = st.selectbox("Karşılaştırmak için ikinci sütun", numeric_cols, index=1)
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col], y=df[other_col], ax=ax)
            st.pyplot(fig)

        with tab4:
            st.markdown("### X̄-R Kontrol Grafiği")
            subgroup_size = st.slider("Alt grup büyüklüğü", min_value=2, max_value=10, value=5)
            num_subgroups = len(df[col]) // subgroup_size
            subgroups = np.array_split(df[col].dropna().values[:num_subgroups * subgroup_size], num_subgroups)
            x_bar = [np.mean(sg) for sg in subgroups]
            r_bar = [np.ptp(sg) for sg in subgroups]
            overall_mean = np.mean(x_bar)
            overall_range = np.mean(r_bar)
            UCL = overall_mean + 3 * (overall_range / subgroup_size ** 0.5)
            LCL = overall_mean - 3 * (overall_range / subgroup_size ** 0.5)

            fig, ax = plt.subplots()
            ax.plot(x_bar, marker='o', label='X̄')
            ax.axhline(overall_mean, color='green', linestyle='--', label='Ortalama')
            ax.axhline(UCL, color='red', linestyle='--', label='UCL')
            ax.axhline(LCL, color='red', linestyle='--', label='LCL')
            ax.set_title('X̄ Kontrol Grafiği')
            ax.legend()
            st.pyplot(fig)

        st.subheader("İstatistiksel Testler")
        test_type = st.selectbox("Test seçin", ["T-Testi", "ANOVA"])

        if test_type == "T-Testi":
            group_col = st.selectbox("Gruplandırma için kategorik sütun seçin", df.select_dtypes(include='object').columns)
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                group1 = df[df[group_col] == groups[0]][col].dropna()
                group2 = df[df[group_col] == groups[1]][col].dropna()
                t_stat, p_val = stats.ttest_ind(group1, group2)
                st.write(f"T-istatistiği: {t_stat:.4f}, p-değeri: {p_val:.4f}")
            else:
                st.warning("T-testi için sadece 2 grup olmalı.")

        elif test_type == "ANOVA":
            group_col = st.selectbox("Gruplandırma için kategorik sütun seçin", df.select_dtypes(include='object').columns, key="anova")
            anova_groups = [group[col].dropna() for name, group in df.groupby(group_col)]
            f_stat, p_val = stats.f_oneway(*anova_groups)
            st.write(f"F-istatistiği: {f_stat:.4f}, p-değeri: {p_val:.4f}")
    else:
        st.warning("Sayısal sütun bulunamadı.")
else:
    st.info("Başlamak için bir CSV dosyası yükleyin.")
