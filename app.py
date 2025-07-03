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

# ======================= INISIALISASI ========================
if "menu" not in st.session_state:
    st.session_state["menu"] = "🏠 Beranda"

# ======================= HEADER DAN LAYOUT ========================
st.markdown("<h1 style='text-align: center;'>📊 Aplikasi Klastering Potensi Ekonomi</h1>", unsafe_allow_html=True)
col_menu, col_main = st.columns([1, 4], gap="medium")

# ======================= MENU DI KIRI ========================
with col_menu:
    st.markdown("""
        <style>
        .menu-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .menu-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #222;
            display: flex;
            align-items: center;
        }
        .menu-title::before {
            content: "🚀";
            margin-right: 10px;
            font-size: 22px;
        }
        .menu-button {
            display: block;
            width: 100%;
            padding: 12px 16px;
            margin-bottom: 12px;
            font-size: 16px;
            font-weight: 500;
            text-align: left;
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 10px;
            transition: background-color 0.2s ease;
        }
        .menu-button:hover {
            background-color: #e9f0ff;
            cursor: pointer;
        }
        .info-box {
            background-color: #e7f1ff;
            border-left: 5px solid #4a90e2;
            padding: 16px;
            border-radius: 10px;
            font-size: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
    # Menu
    st.markdown('<div class="menu-box">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Navigasi</div>', unsafe_allow_html=True)
    if st.button("🏠 Beranda", key="btn1"): st.session_state["menu"] = "🏠 Beranda"
    if st.button("📊 Klastering Data Internal", key="btn2"): st.session_state["menu"] = "🧮 Klastering Data Internal"
    if st.button("🗺️ Visualisasi Peta", key="btn3"): st.session_state["menu"] = "🗺️ Visualisasi Peta"
    if st.button("📤 Unggah Data Sendiri", key="btn4"): st.session_state["menu"] = "📤 Unggah Data Sendiri"
    st.markdown('</div>', unsafe_allow_html=True)

    # Tentang
    st.markdown('### ℹ️ Tentang')
    st.markdown("""
    <div class="info-box">
    Aplikasi untuk memetakan potensi ekonomi wilayah di Sulawesi Tenggara menggunakan 
    <b>Fuzzy C-Means Clustering</b>.<br><br>
    </div>
    """, unsafe_allow_html=True)


menu = st.session_state["menu"]

# ======================= KONTEN UTAMA ========================
with col_main:
    if menu == "🏠 Beranda":
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
            - Evaluasi klaster menggunakan metrik seperti Silhouette Score
            - Dukungan data internal dan upload data mandiri

            🚀 Silakan mulai dari menu di sebelah kiri untuk menjelajahi fitur-fitur aplikasi ini!
            """)
        st.markdown("---")
        st.markdown("<p style='text-align: center;'>© 2025</p>", unsafe_allow_html=True)

    elif menu == "🧮 Klastering Data Internal":
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

            st.subheader("📈 Evaluasi Klaster")
            st.markdown(f"""
            - **Silhouette Score**: {silhouette_score(X_scaled, fcm_labels):.4f}  
            - **Davies-Bouldin Index**: {davies_bouldin_score(X_scaled, fcm_labels):.4f}  
            - **Calinski-Harabasz Score**: {calinski_harabasz_score(X_scaled, fcm_labels):.2f}
            """)

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

            st.subheader("📋 Tabel Hasil")
            st.dataframe(df)
            csv = df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Unduh Hasil CSV", data=csv, file_name=f"hasil_fcm_{selected_sheet}.csv", mime="text/csv")

            st.session_state["df_klaster"] = df
            st.session_state["selected_sheet"] = selected_sheet

        except Exception as e:
            st.error(f"Gagal memproses data: {e}")

    # ======================= INTERPRETASI FUNGSIONAL ========================
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

# ======================= VISUALISASI PETA ========================
with col_main:
    if menu == "🗺️ Visualisasi Peta":
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

                # ===== Tampilan berdampingan =====
                col_peta, col_interpretasi = st.columns([2, 1], gap="large")

                with col_peta:
                    st_folium(m, width=550, height=500)

                with col_interpretasi:
                    st.markdown("### 📖 Interpretasi Klaster")
                    sheet_name = st.session_state.get("selected_sheet", "")
                    st.markdown(interpretasi_klaster(sheet_name), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gagal memuat peta: {e}")


    elif menu == "📤 Unggah Data Sendiri":
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

                st.subheader("📈 Evaluasi Klaster")
                st.markdown(f"""
                - **Silhouette Score**: {silhouette_score(X_scaled, fcm_labels):.4f}  
                - **Davies-Bouldin Index**: {davies_bouldin_score(X_scaled, fcm_labels):.4f}  
                - **Calinski-Harabasz Score**: {calinski_harabasz_score(X_scaled, fcm_labels):.2f}
                """)

                # Visualisasi PCA
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_scaled)
                fig1, ax1 = plt.subplots()
                for i in range(k):
                    ax1.scatter(components[fcm_labels == i, 0], components[fcm_labels == i, 1], label=f"Cluster {i}", alpha=0.7)
                ax1.set_title("Visualisasi Klaster (PCA)")
                ax1.legend()
                st.pyplot(fig1)

                # Matriks Keanggotaan
                membership_df = pd.DataFrame(u.T, index=df.index, columns=[f'Cluster_{i}' for i in range(k)])
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.heatmap(membership_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
                ax2.set_title("Matriks Keanggotaan Fuzzy")
                st.pyplot(fig2)

                # Distribusi klaster
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
