import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard IKK Nasional",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏗️ Dashboard Indeks Kemahalan Konstruksi (IKK)")
st.markdown("Estimasi biaya konstruksi berbasis data BPS **2014–2025** · *Linear Regression per Provinsi*")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — UPLOAD FILE
# ─────────────────────────────────────────────
st.sidebar.header("📂 Upload Data CSV")
st.sidebar.markdown(
    "Upload **satu atau lebih** file CSV IKK dari BPS.\n\n"
    "Format nama file yang diharapkan:\n"
    "`Indeks Kemahalan Konstruksi, YYYY.csv`\n\n"
    "Rentang tahun yang didukung: **2014–2025**"
)

uploaded_files = st.sidebar.file_uploader(
    "Pilih file CSV (bisa banyak sekaligus)",
    type=["csv"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Parameter Simulasi**")
luas_input = st.sidebar.number_input(
    "Luas Bangunan (m²)", min_value=10, max_value=10000, value=100, step=10
)
harga_input = st.sidebar.number_input(
    "Harga Dasar per m² (Rp)",
    min_value=1_000_000, max_value=50_000_000,
    value=5_000_000, step=500_000, format="%d"
)

# ─────────────────────────────────────────────
# FUNGSI: BACA & BERSIHKAN DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(files_bytes_list, luas, harga_dasar):
    all_years_data = []

    for file_name, file_bytes in files_bytes_list:
        match = re.search(r'(20\d{2})', file_name)
        if not match:
            st.sidebar.warning(f"⚠ Tahun tidak terdeteksi: {file_name}")
            continue
        tahun = int(match.group(1))
        if tahun < 2014 or tahun > 2025:
            st.sidebar.warning(f"⚠ Tahun {tahun} di luar rentang 2014–2025: {file_name}")
            continue

        try:
            df_temp = pd.read_csv(
                io.BytesIO(file_bytes),
                skiprows=3,
                names=['provinsi', 'ikk'],
                sep=None,
                engine='python'
            )
        except Exception as e:
            st.sidebar.error(f"Gagal membaca {file_name}: {e}")
            continue

        df_temp['provinsi'] = (
            df_temp['provinsi'].astype(str)
            .str.upper().str.strip()
            .str.replace(r'[\r\n\t]+', ' ', regex=True).str.strip()
        )
        df_temp['ikk'] = (
            df_temp['ikk'].astype(str).str.replace(',', '.', regex=False)
        )
        df_temp['ikk'] = pd.to_numeric(df_temp['ikk'], errors='coerce')
        df_temp.dropna(subset=['provinsi', 'ikk'], inplace=True)
        df_temp = df_temp[~df_temp['provinsi'].str.contains('INDONESIA', na=False)]
        df_temp = df_temp[df_temp['provinsi'].str.strip() != '']
        df_temp['tahun'] = tahun
        all_years_data.append(df_temp)

    if not all_years_data:
        return pd.DataFrame()

    df = pd.concat(all_years_data, ignore_index=True)
    df['estimasi_biaya'] = luas * harga_dasar * (df['ikk'] / 100)
    return df


# ─────────────────────────────────────────────
# FUNGSI: LATIH MODEL PER PROVINSI
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_models(df):
    models = {}
    evaluasi = []

    for prov in df['provinsi'].unique():
        df_prov = df[df['provinsi'] == prov].sort_values('tahun')
        if len(df_prov) < 2:
            continue

        X = df_prov[['tahun']].values
        y = df_prov['ikk'].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        try:
            r2 = r2_score(y, y_pred)
        except Exception:
            r2 = 0.0

        models[prov] = model
        evaluasi.append({
            'Provinsi': prov,
            'Intercept': round(model.intercept_, 2),
            'Slope (Tren/Tahun)': round(model.coef_[0], 4),
            'R²': round(r2, 4),
            'Jumlah Data (Tahun)': len(df_prov)
        })

    return models, pd.DataFrame(evaluasi)


# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if not uploaded_files:
    st.info("👈 Upload file CSV IKK BPS melalui sidebar kiri untuk memulai analisis.")
    st.markdown("""
    **Cara penggunaan:**
    1. Siapkan file CSV IKK dari BPS (tahun 2014–2025)
    2. Upload lewat panel sidebar kiri (bisa sekaligus banyak file)
    3. Eksplorasi tren, prediksi, dan evaluasi model di tab yang tersedia
    """)
    st.stop()

# Baca bytes dari semua file
files_data = [(f.name, f.read()) for f in uploaded_files]

with st.spinner("Memuat dan memproses data CSV…"):
    df = load_data(tuple(files_data), luas_input, harga_input)

if df.empty:
    st.error("❌ Data kosong. Periksa format dan nama file CSV Anda.")
    st.stop()

with st.spinner("Melatih model regresi per provinsi…"):
    models, df_eval = train_models(df)

# ─────────────────────────────────────────────
# METRIK RINGKASAN
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Rentang Tahun",
            f"{int(df['tahun'].min())}–{int(df['tahun'].max())}")
col2.metric("🗺️ Jumlah Provinsi", df['provinsi'].nunique())
col3.metric("📋 Total Baris Data", f"{len(df):,}")
col4.metric("🤖 Model Terlatih", len(models))

st.markdown("---")

# ─────────────────────────────────────────────
# TAB NAVIGASI
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tren IKK",
    "🔮 Prediksi & Estimasi Biaya",
    "🗃️ Data Lengkap",
    "📊 Evaluasi Model"
])

# ══════════════════════════════════════════════
# TAB 1 — TREN IKK
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Tren Historis IKK per Provinsi (2014–2025)")

    semua_provinsi = sorted(df['provinsi'].unique())
    default_prov = [p for p in ['DKI JAKARTA', 'JAWA BARAT', 'ACEH', 'KALIMANTAN TIMUR', 'PAPUA']
                    if p in semua_provinsi]

    provinsi_dipilih = st.multiselect(
        "Pilih provinsi yang ingin dibandingkan:",
        options=semua_provinsi,
        default=default_prov[:5]
    )

    if provinsi_dipilih:
        df_vis = df[df['provinsi'].isin(provinsi_dipilih)]
        tahun_min = int(df['tahun'].min())
        tahun_max = int(df['tahun'].max())

        fig, ax = plt.subplots(figsize=(13, 5))
        sns.lineplot(data=df_vis, x='tahun', y='ikk', hue='provinsi',
                     marker='o', linewidth=2.5, ax=ax)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Basis Nasional (100)')
        ax.set_title(f'Tren IKK Provinsi Terpilih ({tahun_min}–{tahun_max})', fontsize=14)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Nilai IKK (Basis = 100)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title='Provinsi', bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.set_xticks(range(tahun_min, tahun_max + 1))
        plt.tight_layout()
        st.pyplot(fig)

        # Tabel ringkasan
        df_pivot = df_vis.pivot_table(index='provinsi', columns='tahun', values='ikk').round(2)
        st.markdown("**Tabel Nilai IKK:**")
        st.dataframe(df_pivot, use_container_width=True)
    else:
        st.warning("Pilih minimal satu provinsi dari dropdown di atas.")

# ══════════════════════════════════════════════
# TAB 2 — PREDIKSI & ESTIMASI BIAYA
# ══════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Prediksi IKK & Estimasi Biaya Konstruksi")
    st.caption("Parameter luas dan harga dasar bisa diubah di sidebar kiri.")

    col_a, col_b = st.columns(2)
    with col_a:
        prov_pred = st.selectbox("Pilih Provinsi", options=sorted(models.keys()))
    with col_b:
        tahun_pred = st.number_input(
            "Tahun Prediksi", min_value=2010, max_value=2040, value=2026, step=1
        )

    if st.button("🚀 Hitung Prediksi", use_container_width=True):
        model_sel = models[prov_pred]
        ikk_pred = model_sel.predict([[tahun_pred]])[0]
        biaya_pred = luas_input * harga_input * (ikk_pred / 100)

        r1, r2_col, r3, r4 = st.columns(4)
        r1.metric("📐 IKK Prediksi", f"{ikk_pred:.2f}")
        r2_col.metric("💰 Estimasi Biaya", f"Rp {biaya_pred:,.0f}")
        r3.metric("📏 Luas", f"{luas_input} m²")
        r4.metric("📍 Provinsi", prov_pred)

        # Grafik historis + proyeksi
        df_hist = df[df['provinsi'] == prov_pred].sort_values('tahun')
        tahun_hist_min = int(df_hist['tahun'].min())
        tahun_hist_max = int(df_hist['tahun'].max())

        # Garis regresi model di rentang historis
        tahun_fit = list(range(tahun_hist_min, tahun_hist_max + 1))
        ikk_fit   = [model_sel.predict([[t]])[0] for t in tahun_fit]

        # Proyeksi ke depan (hanya jika tahun_pred > data terakhir)
        if tahun_pred > tahun_hist_max:
            tahun_proj = list(range(tahun_hist_max + 1, tahun_pred + 1))
            ikk_proj   = [model_sel.predict([[t]])[0] for t in tahun_proj]
        else:
            tahun_proj, ikk_proj = [], []

        fig2, ax2 = plt.subplots(figsize=(11, 4))

        # Data asli
        ax2.plot(df_hist['tahun'], df_hist['ikk'],
                 marker='o', color='steelblue', label='Data Historis', linewidth=2.5, zorder=3)

        # Garis regresi di rentang historis
        ax2.plot(tahun_fit, ikk_fit,
                 linestyle='--', color='orange', alpha=0.7, label='Garis Regresi', linewidth=1.5)

        # Proyeksi masa depan
        if tahun_proj:
            ax2.plot(tahun_proj, ikk_proj,
                     marker='s', linestyle='--', color='tomato', label='Proyeksi', linewidth=2)

        # Titik target prediksi
        ax2.scatter([tahun_pred], [ikk_pred], color='tomato', s=140, zorder=5)
        ax2.annotate(
            f'  IKK {tahun_pred} = {ikk_pred:.2f}',
            xy=(tahun_pred, ikk_pred),
            fontsize=11, color='tomato', va='bottom'
        )
        ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.4, label='Basis Nasional (100)')
        ax2.set_title(f'Tren & Proyeksi IKK — {prov_pred}', fontsize=13)
        ax2.set_xlabel('Tahun')
        ax2.set_ylabel('IKK')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2)

        st.info(
            f"💡 Dengan IKK prediksi **{ikk_pred:.2f}**, biaya konstruksi rumah "
            f"**{luas_input} m²** di **{prov_pred}** pada tahun **{tahun_pred}** "
            f"diperkirakan sekitar **Rp {biaya_pred:,.0f}**"
            f" (harga dasar Rp {harga_input:,}/m²)."
        )

# ══════════════════════════════════════════════
# TAB 3 — DATA LENGKAP
# ══════════════════════════════════════════════
with tab3:
    st.subheader("🗃️ Data IKK Lengkap")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_prov = st.multiselect(
            "Filter Provinsi:", options=sorted(df['provinsi'].unique()), default=[]
        )
    with col_f2:
        tahun_tersedia = sorted(df['tahun'].unique())
        filter_tahun = st.multiselect(
            "Filter Tahun:", options=tahun_tersedia, default=[]
        )

    df_tampil = df.copy()
    if filter_prov:
        df_tampil = df_tampil[df_tampil['provinsi'].isin(filter_prov)]
    if filter_tahun:
        df_tampil = df_tampil[df_tampil['tahun'].isin(filter_tahun)]

    df_tampil = df_tampil.sort_values(['provinsi', 'tahun']).reset_index(drop=True)
    df_tampil.columns = ['Provinsi', 'IKK', 'Tahun', 'Estimasi Biaya (Rp)']

    st.dataframe(df_tampil, use_container_width=True, height=450)
    st.caption(f"Menampilkan {len(df_tampil):,} dari {len(df):,} baris data.")

    csv_export = df_tampil.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇ Download Data (CSV)",
        data=csv_export,
        file_name="data_ikk_filtered.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════
# TAB 4 — EVALUASI MODEL
# ══════════════════════════════════════════════
with tab4:
    st.subheader("📊 Performa Model Linear Regression per Provinsi")

    if not df_eval.empty:
        col_sort = st.selectbox("Urutkan berdasarkan:", ['R²', 'Slope (Tren/Tahun)', 'Provinsi'])
        asc = col_sort == 'Provinsi'
        df_show = df_eval.sort_values(col_sort, ascending=asc).reset_index(drop=True)

        st.dataframe(df_show, use_container_width=True, height=420)

        fig3, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(df_eval['R²'], bins=15, color='steelblue', edgecolor='white', alpha=0.85)
        axes[0].set_title('Distribusi R² Seluruh Model')
        axes[0].set_xlabel('R² Score')
        axes[0].set_ylabel('Frekuensi')
        axes[0].grid(True, linestyle='--', alpha=0.4)

        axes[1].hist(df_eval['Slope (Tren/Tahun)'], bins=15, color='tomato', edgecolor='white', alpha=0.85)
        axes[1].set_title('Distribusi Slope (Tren per Tahun)')
        axes[1].set_xlabel('Slope')
        axes[1].set_ylabel('Frekuensi')
        axes[1].grid(True, linestyle='--', alpha=0.4)
        axes[1].axvline(x=0, color='gray', linestyle='--')

        plt.tight_layout()
        st.pyplot(fig3)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rata-rata R²", f"{df_eval['R²'].mean():.4f}")
        c2.metric("Median R²", f"{df_eval['R²'].median():.4f}")
        c3.metric("Total Model", len(df_eval))
    else:
        st.warning("Belum ada model yang dievaluasi.")

st.markdown("---")
st.caption("Dashboard IKK Nasional · Data BPS Indonesia 2014–2025 · Dibuat dengan Streamlit")
