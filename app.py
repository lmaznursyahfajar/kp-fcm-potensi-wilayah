import streamlit as st

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi FCM", layout="wide")

# Import pustaka
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ======================= CSS ========================
st.markdown("""
<style>
.sidebar-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 5px;
    color: orange;
    font-weight: normal;
    cursor: pointer;
    transition: all 0.2s ease;
}
.sidebar-item:hover {
    background-color: #e9ecef;
}
.sidebar-item.active {
    background-color: green;
    color: white !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ======================= MENU ========================
menu_dict = {
    "🏠 Beranda": "Beranda",
    "🧮 Klastering Data Internal": "Klastering",
    "🗺️ Visualisasi Peta": "Peta",
    "📌 Rekomendasi Kebijakan": "Rekomendasi",
    "📤 Unggah Data Sendiri": "Unggah"
}

# ======================= INISIALISASI ========================
if "selected_label" not in st.session_state:
    st.session_state.selected_label = list(menu_dict.keys())[0]

# ======================= SIDEBAR ========================
with st.sidebar:
    st.markdown("<div class='sidebar-custom'>", unsafe_allow_html=True)

    for label in menu_dict:
        active = "active" if label == st.session_state.selected_label else ""

        # Jika menu ini dipilih, tampilkan styled saja (tidak perlu button)
        if label == st.session_state.selected_label:
            st.markdown(
                f"""
                <div class="sidebar-item {active}">
                    <span class="icon">{label.split()[0]}</span>
                    <span>{' '.join(label.split()[1:])}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Jika menu lain, buat tombol yang akan mengubah session_state
            if st.button(label, key=f"menu_{label}"):
                st.session_state.selected_label = label
                st.rerun()  # agar langsung berpindah halaman

# ======================= LOGIKA PILIHAN MENU ========================
menu = menu_dict[st.session_state.selected_label]
# ======================= JUDUL UTAMA ========================
st.markdown("<h1 style='text-align: center;'>📊 Aplikasi Klastering Potensi Ekonomi</h1>", unsafe_allow_html=True)

def interpretasi_klaster(sheet_name):
    if "pertanian" in sheet_name.lower():
        return """
        #### 🌾 Interpretasi Klaster (Pertanian)
        - **Klaster 0**:  Sentra Produksi Padi & Jagung Skala Besar.
        - **Klaster 1**: Wilayah Non-Agraris atau Pendukung. Lahan kecil atau sedikit komoditas.
        - **Klaster 2**: Pertanian Menengah & Diversifikasi Komoditas.
        """
    elif "umkm" in sheet_name.lower():
        return """
        #### 🛍️ Interpretasi Klaster (UMKM)
        - **Klaster 0**: Dominasi Usaha Skala Menengah & Besar.
        - **Klaster 1**: Aktivitas UMKM Rendah.
        - **Klaster 2**: Dominasi Usaha Mikro & Kecil.
        """
    elif "perkebunan" in sheet_name.lower() and "buah" not in sheet_name.lower():
        return """
        #### 🌴 Interpretasi Klaster (Perkebunan)
        - **Klaster 0**: Sentra Perkebunan Kakao dan Komoditas Ekspor.
        - **Klaster 1**: Wilayah Perkebunan Kecil & Skala Rumah Tangga.
        - **Klaster 2**: Sentra Produksi Menengah dengan Wilayah Campuran dengan Variasi Komoditas.
        """
    elif "perkebunan_buah" in sheet_name.lower():
        return """
        #### 🍍 Interpretasi Klaster (Perkebunan Buah)
        - **Klaster 0**: Produksi Kecil dan Menengah.
        - **Klaster 1**: Produksi Besar dan Terdiversifikasi.
        - **Klaster 2**: Pusat Sentra Unggulan.
        """
    elif "perikanan" in sheet_name.lower():
        return """
        #### 🐟 Interpretasi Klaster (Perikanan)
        - **Klaster 0**: Wilayah Fokus Budidaya Skala Menengah.
        - **Klaster 1**: Wilayah dengan Potensi Budidaya Skala Besar.
        - **Klaster 2**: Wilayah Dominan Perikanan Tangkap.
        """
    else:
        return "⚠️ Data tidak termasuk dalam kategori pertanian, perikanan, atau perkebunan."

# ======================= MENU BERANDA ========================
if menu == "Beranda":
    st.markdown("<h4 style='text-align: center;'>Provinsi Sulawesi Tenggara</h4>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("beranda.jpg", caption="Potensi Wilayah", use_container_width=True)
    with col2:
        st.markdown("""
        Selamat datang di **Aplikasi Klastering Fuzzy C-Means**, sebuah aplikasi analisis spasial untuk mengelompokkan 
        potensi ekonomi sektor primer di wilayah Provinsi Sulawesi Tenggara. 

        Aplikasi ini dikembangkan menggunakan metode **Fuzzy C-Means (FCM)** untuk membantu pengambil kebijakan, 
        akademisi, maupun masyarakat umum dalam memahami struktur potensi wilayah berbasis data.

        ### 🎯 Fitur Utama:
        - Klasterisasi berdasarkan data pertanian, perikanan, perkebunan, dll
        - Visualisasi hasil dalam bentuk grafik dan peta interaktif
        - Dukungan data internal dan upload data mandiri

        🚀 Silakan mulai dari menu di sebelah kiri untuk menjelajahi fitur-fitur aplikasi ini!
        """)
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>© 2025</p>", unsafe_allow_html=True)

# ======================= MENU KLASTERING ========================
elif menu == "Klastering":
    st.title("🔍 Klastering Data Internal (FCM)")
    try:
        file_path = "data_sultra.xlsx"
        sheet_names = pd.ExcelFile(file_path).sheet_names
        selected_sheet = st.selectbox("Pilih jenis data:", sheet_names)

        df = pd.read_excel(file_path, sheet_name=selected_sheet)
        df = df.rename(columns={df.columns[0]: "Kabupaten/Kota"}).set_index("Kabupaten/Kota").fillna(0)

        st.subheader("📄 Data Awal")
        st.dataframe(df)

        k = st.slider("Jumlah Klaster", 2, 10, 3)
        m_param = 2

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        np.random.seed(42)
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_scaled.T, c=k, m=m_param, error=0.005, maxiter=1000)
        fcm_labels = np.argmax(u, axis=0)
        df["FCM_Cluster"] = fcm_labels

        st.subheader("📊 Visualisasi Klaster")

        col_pca, col_matriks, col_distribusi = st.columns(3)

        with col_pca:
            st.markdown("**📌 PCA**")
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            fig1, ax1 = plt.subplots()
            for i in range(k):
                ax1.scatter(components[fcm_labels == i, 0], components[fcm_labels == i, 1], label=f"Cluster {i}", alpha=0.7)
            ax1.set_title("PCA")
            ax1.legend()
            st.pyplot(fig1)

        with col_matriks:
            st.markdown("**📌 Matriks Keanggotaan**")
            membership_df = pd.DataFrame(u.T, index=df.index, columns=[f'Cluster_{i}' for i in range(k)])
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(membership_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2, cbar=False)
            ax2.set_title("Matriks Keanggotaan")
            st.pyplot(fig2)

        with col_distribusi:
            st.markdown("**📌 Distribusi Klaster**")
            cluster_counts = df["FCM_Cluster"].value_counts().sort_index()
            fig3, ax3 = plt.subplots()
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis', ax=ax3)
            ax3.set_title("Distribusi Wilayah")
            st.pyplot(fig3)

        st.subheader("📋 Tabel Hasil")
        st.dataframe(df)

        csv = df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh Hasil CSV", data=csv, file_name=f"hasil_fcm_{selected_sheet}.csv", mime="text/csv")

        st.session_state["df_klaster"] = df
        st.session_state["selected_sheet"] = selected_sheet

    except Exception as e:
        st.error(f"Gagal memproses data: {e}")

# ======================= MENU VISUALISASI PETA ========================
elif menu == "Peta":
    st.title("🗺️ Visualisasi Peta")
    try:
        df_klaster = st.session_state.get("df_klaster")
        if df_klaster is None:
            st.warning("⚠️ Jalankan klastering pada data internal terlebih dahulu.")
        else:
            shp_path = "sultra.shp"
            gdf = gpd.read_file(shp_path)

            gdf["kab_kota"] = gdf["kab_kota"].str.strip().str.lower()
            df_klaster = df_klaster.rename_axis("kab_kota").reset_index()
            df_klaster["kab_kota"] = df_klaster["kab_kota"].str.strip().str.lower()

            gdf = gdf.set_index("kab_kota").join(
                df_klaster.set_index("kab_kota")[["FCM_Cluster"]], how="left"
            ).reset_index()

            missing = gdf[gdf["FCM_Cluster"].isna()]
            if not missing.empty:
                st.warning(f"❗ Wilayah berikut tidak ditemukan dalam hasil klastering:\n{missing['kab_kota'].tolist()}")

            k = df_klaster["FCM_Cluster"].nunique()
            colors = ['red', 'yellow', 'green', 'blue', 'purple', 'orange', 'pink', 'cyan', 'lime', 'brown']
            color_map = {i: colors[i % len(colors)] for i in range(k)}

            def style_function(feature):
                cluster = feature['properties']['FCM_Cluster']
                return {
                    'fillColor': color_map.get(cluster, 'gray'),
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.6
                }

            m = folium.Map(location=[-4.0, 122.5], zoom_start=8)
            folium.GeoJson(
                gdf,
                name="Klaster",
                tooltip=folium.GeoJsonTooltip(fields=["kab_kota", "FCM_Cluster"],
                                              aliases=["Kabupaten/Kota", "Klaster"]),
                style_function=style_function
            ).add_to(m)
            folium.LayerControl().add_to(m)

            col_peta, col_interpretasi = st.columns([2, 1], gap="large")
            with col_peta:
                st_folium(m, width=550, height=500)
            with col_interpretasi:
                st.markdown("### 📖 Interpretasi Klaster")
                sheet_name = st.session_state.get("selected_sheet", "")
                st.markdown(interpretasi_klaster(sheet_name), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Gagal memuat peta: {e}")

# ======================= MENU REKOMENDASI KEBIJAKAN ========================
elif menu == "Rekomendasi":
    st.title("📑 Sistem Rekomendasi Kebijakan")

    df_klaster = st.session_state.get("df_klaster")
    selected_sheet = st.session_state.get("selected_sheet", "")

    if df_klaster is None:
        st.warning("⚠️ Silakan jalankan klastering data internal terlebih dahulu.")
    else:
        st.subheader(f"📊 Rekomendasi Berdasarkan Klaster - {selected_sheet.title()}")

        def rekomendasi_kebijakan(sheet, klaster):
            sheet = sheet.lower()
            if "pertanian" in sheet:
                if klaster == 0:
                    return "💧 Fokuskan pada perbaikan irigasi, penyuluhan teknologi pertanian, dan hilirisasi hasil padi-jagung."
                elif klaster == 1:
                    return "🌱 Program intensifikasi lahan kecil & pelatihan pertanian berkelanjutan untuk rumah tangga tani."
                else:
                    return "🌾 Kembangkan komoditas alternatif (hortikultura/umbi) dan fasilitasi pasar lokal."
            elif "umkm" in sheet:
                if klaster == 0:
                    return "📦 Dorong ekspor UMKM dan adopsi teknologi digital untuk skala produksi besar."
                elif klaster == 1:
                    return "📘 Perluas literasi bisnis, kemudahan akses permodalan, dan dukungan regulasi."
                else:
                    return "🧵 Bentuk inkubator UMKM baru dan fasilitasi pelatihan kewirausahaan dasar."
            elif "perkebunan_buah" in sheet:
                if klaster == 0:
                    return "🍋 Bangun koperasi tani buah dan fasilitasi akses pupuk & bibit unggul."
                elif klaster == 1:
                    return "🍍 Kembangkan jalur distribusi hasil panen dan pengolahan buah skala industri."
                else:
                    return "🏭 Prioritaskan pengolahan pascapanen dan branding buah unggulan daerah."
            elif "perkebunan" in sheet:
                if klaster == 0:
                    return "☕ Dorong hilirisasi kakao & kopi serta penguatan koperasi ekspor."
                elif klaster == 1:
                    return "🌿 Program revitalisasi perkebunan rakyat dan akses kredit kecil."
                else:
                    return "🧪 Diversifikasi komoditas dan pembinaan agribisnis berkelanjutan."
            elif "perikanan" in sheet:
                if klaster == 0:
                    return "🐟 Perkuat budidaya dengan teknologi pakan, bibit unggul, dan kemitraan pasar."
                elif klaster == 1:
                    return "🌊 Modernisasi tambak dan ekspansi produksi berbasis ekspor."
                else:
                    return "⚓ Dukung nelayan kecil dengan alat tangkap modern dan subsidi solar nelayan."
            else:
                return "🔍 Belum tersedia rekomendasi untuk sektor ini."

        for idx, row in df_klaster.iterrows():
            kab = idx
            klaster = row["FCM_Cluster"]
            rekom = rekomendasi_kebijakan(selected_sheet, klaster)
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                <strong>📍 {kab.title()} (Klaster {klaster})</strong><br>
                {rekom}
            </div>
            """, unsafe_allow_html=True)

# ======================= MENU UPLOAD DATA SENDIRI ========================
elif menu == "Unggah":
    st.title("🧾 Klastering dari Data Unggahan")
    uploaded_file = st.file_uploader("📤 Unggah File Excel (.xlsx)", type="xlsx")

    if uploaded_file:
        try:
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names
            selected_sheet = st.selectbox("Pilih Sheet:", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            df = df.rename(columns={df.columns[0]: "Wilayah"}).set_index("Wilayah").fillna(0)

            st.subheader("📄 Data Awal")
            st.dataframe(df)

            k = st.slider("Jumlah Klaster", 2, 10, 3, key="cluster_slider_2")
            m_param = 2

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df)
            np.random.seed(42)
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_scaled.T, c=k, m=m_param, error=0.005, maxiter=1000)
            fcm_labels = np.argmax(u, axis=0)
            df["FCM_Cluster"] = fcm_labels

            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            fig1, ax1 = plt.subplots()
            for i in range(k):
                ax1.scatter(components[fcm_labels == i, 0], components[fcm_labels == i, 1], label=f"Cluster {i}", alpha=0.7)
            ax1.set_title("Visualisasi Klaster (PCA)")
            ax1.legend()
            st.pyplot(fig1)

            membership_df = pd.DataFrame(u.T, index=df.index, columns=[f'Cluster_{i}' for i in range(k)])
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(membership_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
            ax2.set_title("Matriks Keanggotaan Fuzzy")
            st.pyplot(fig2)

            cluster_counts = df["FCM_Cluster"].value_counts().sort_index()
            fig3, ax3 = plt.subplots()
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis', ax=ax3)
            ax3.set_title("Distribusi Wilayah per Klaster")
            st.pyplot(fig3)

            st.subheader("📋 Hasil Klastering")
            st.dataframe(df)

            csv = df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Unduh Hasil CSV", data=csv, file_name="hasil_fcm_upload.csv", mime="text/csv")

            csv_membership = membership_df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Unduh Matriks Keanggotaan", data=csv_membership, file_name="membership_fcm_upload.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")
