import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings

from streamlit_option_menu import option_menu

# Menonaktifkan warnings
warnings.filterwarnings("ignore")

# Load dataset
hour_df = pd.read_csv("../data/hour.csv")
day_df = pd.read_csv("../data/day.csv")

# Outlier
Q1 = hour_df["cnt"].quantile(0.25)
Q3 = hour_df["cnt"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = hour_df[(hour_df["cnt"] < lower_bound) | (hour_df["cnt"] > upper_bound)]
hour_df = hour_df[(hour_df["cnt"] >= lower_bound) & (hour_df["cnt"] <= upper_bound)]

# Konversi kolom "dteday" ke tipe datetime
hour_df["dteday"] = pd.to_datetime(hour_df["dteday"])
day_df["dteday"] = pd.to_datetime(day_df["dteday"])

# Buat time-based features baru
hour_df['year'] = hour_df['dteday'].dt.year
hour_df['month'] = hour_df['dteday'].dt.month
hour_df['day'] = hour_df['dteday'].dt.day
hour_df['day_of_week'] = hour_df['dteday'].dt.day_name()
hour_df['hour_of_day'] = hour_df['hr']

day_df['year'] = day_df['dteday'].dt.year
day_df['month'] = day_df['dteday'].dt.month
day_df['day'] = day_df['dteday'].dt.day
day_df['day_of_week'] = day_df['dteday'].dt.day_name()

# Mapping season
season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
hour_df['season_name'] = hour_df['season'].map(season_map)
day_df['season_name'] = day_df['season'].map(season_map)

# Mapping weather
weather_map = {
    1: 'Clear/Partly Cloudy',
    2: 'Mist/Cloudy',
    3: 'Light Precipitation',
    4: 'Heavy Precipitation'
}
hour_df['weather_condition'] = hour_df['weathersit'].map(weather_map)
day_df['weather_condition'] = day_df['weathersit'].map(weather_map)

# Persentase dari casual vs registered users
hour_df['casual_pct'] = hour_df['casual'] / hour_df['cnt'] * 100
hour_df['registered_pct'] = hour_df['registered'] / hour_df['cnt'] * 100

day_df['casual_pct'] = day_df['casual'] / day_df['cnt'] * 100
day_df['registered_pct'] = day_df['registered'] / day_df['cnt'] * 100

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Home", "Dataset Overview", "Visualization & Explanatory", "Clustering", "Conclusion"],
        icons=["house", "table", "bar-chart-line", "bar-chart-line", "check-circle"],
    )

# Halaman "Home"
if selected == "Home":
    st.title("Project: Bike Sharing Analysis")
    st.subheader("Personal info:")
    st.write("""
        - **Nama:** Karinda Amelia
        - **Email :** karindaamelia21@gmail.com
    """)
    
    st.subheader("Analisis akan menjawab pertanyaan berikut:")
    st.write("""
        - Bagaimana variasi jumlah penyewaan sepeda berdasarkan musim, dan musim mana yang memiliki permintaan tertinggi?
        - Bagaimana pengaruh kondisi cuaca terhadap pola penyewaan sepeda?
        - Bagaimana tren penyewaan sepeda per jam sepanjang hari, dan kapan waktu penggunaan tertinggi?
        - Apakah terdapat perbedaan signifikan dalam pola penyewaan sepeda antara hari kerja dan akhir pekan?
        - Bagaimana distribusi dan rasio antara pengguna kasual dan terdaftar di berbagai periode waktu?
        - Bagaimana dampak hari libur terhadap pola penyewaan sepeda dibandingkan dengan hari biasa?
        - Apakah terdapat korelasi antara suhu, kelembaban, kecepatan angin, dan jumlah penyewaan sepeda?
    """)

# Dataset Overview Page
if selected == "Dataset Overview":  
    st.title("Bike Sharing Dataset Overview")
    
    st.subheader("About Dataset")
    st.write(
        "Dataset ini mencakup jumlah penyewaan sepeda secara harian dan per jam antara tahun 2011 dan 2012, "
        "dengan informasi tambahan mengenai musim, cuaca, dan faktor lingkungan."
    )
    
    st.subheader("Attribute Information")
    st.code("""
        - instant: Record index
        - dteday: Tanggal
        - season: Musim (1: Spring, 2: Summer, 3: Fall, 4: Winter)
        - yr: Tahun (0: 2011, 1: 2012)
        - mnth: Bulan (1-12)
        - hr: Jam (0-23, hanya di hour.csv)
        - holiday: Hari libur (1: Ya, 0: Tidak)
        - workingday: Hari kerja (1: Ya, 0: Tidak)
        - weathersit: Kondisi cuaca (1: Cerah, 4: Hujan/Salju)
        - temp: Suhu terukur (skala normalisasi)
        - atemp: Suhu yang dirasakan (skala normalisasi)
        - hum: Kelembaban (0-100)
        - windspeed: Kecepatan angin (0-67)
        - casual: Pengguna kasual
        - registered: Pengguna terdaftar
        - cnt: Total penyewaan sepeda
    """)
    st.write("Sumber data: [Bike Sharing Dataset](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)")

if selected == "Visualization & Explanatory":
    st.title("Visualization & Explanatory Analysis")
    
    # Selectbox untuk memilih pertanyaan analisis
    question = st.selectbox(
        "Pilih pertanyaan analisis:", 
        [
            "Bagaimana variasi jumlah penyewaan sepeda berdasarkan musim, dan musim mana yang memiliki permintaan tertinggi?",
            "Bagaimana pengaruh kondisi cuaca terhadap pola penyewaan sepeda?",
            "Bagaimana tren penyewaan sepeda per jam sepanjang hari, dan kapan waktu penggunaan tertinggi?",
            "Apakah terdapat perbedaan signifikan dalam pola penyewaan sepeda antara hari kerja dan akhir pekan?",
            "Bagaimana distribusi dan rasio antara pengguna kasual dan terdaftar di berbagai periode waktu?",
            "Bagaimana dampak hari libur terhadap pola penyewaan sepeda dibandingkan dengan hari biasa?",
            "Apakah terdapat korelasi antara suhu, kelembaban, kecepatan angin, dan jumlah penyewaan sepeda?"
        ]
    )
    
    # Placeholder untuk menampilkan analisis berdasarkan pertanyaan yang dipilih
    st.subheader(f"**Pertanyaan:**")
    st.write(f"{question}")
    
    # Pertanyaan 1
    if question == "Bagaimana variasi jumlah penyewaan sepeda berdasarkan musim, dan musim mana yang memiliki permintaan tertinggi?":
        
        # Hitung statistik musiman
        seasonal_stats = day_df.groupby("season_name")["cnt"].agg(["mean", "sum"]).sort_values("sum", ascending=False)

        # Buat figure dengan 2 baris dan 2 kolom
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Bar Chart 1: Total Sewa Sepeda Berdasarkan Musim
        sns.barplot(x=seasonal_stats.index, y=seasonal_stats["sum"], palette="viridis", ax=axes[0, 0])
        axes[0, 0].set_title("Total Sewa Sepeda Berdasarkan Musim", fontsize=14)
        axes[0, 0].set_ylabel("Total Sewa (Juta)", fontsize=12)
        axes[0, 0].set_xlabel("Musim", fontsize=12)
        axes[0, 0].yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}")

        for i, v in enumerate(seasonal_stats["sum"]):
            axes[0, 0].text(i, v + 10000, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")
            
        # Bar Chart 2: Rata-rata Sewa Sepeda Harian Berdasarkan Musim
        sns.barplot(x=seasonal_stats.index, y=seasonal_stats["mean"], palette="viridis", ax=axes[0, 1])
        axes[0, 1].set_title("Rata-Rata Sewa Sepeda Harian Berdasarkan Musim", fontsize=14)
        axes[0, 1].set_ylabel("Rata-Rata Sewa per Hari", fontsize=12)
        axes[0, 1].set_xlabel("Musim", fontsize=12)

        for i, v in enumerate(seasonal_stats["mean"]):
            axes[0, 1].text(i, v + 100, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")

        # Line Chart 1: Tren Sewa Sepeda Seiring Waktu Berdasarkan Musim
        sns.lineplot(data=day_df, x="dteday", y="cnt", hue="season_name", palette="viridis", linewidth=2, ax=axes[1, 0])
        axes[1, 0].set_title("Tren Sewa Sepeda Berdasarkan Musim", fontsize=14)
        axes[1, 0].set_ylabel("Total Sewa", fontsize=12)
        axes[1, 0].set_xlabel("Tanggal", fontsize=12)
        axes[1, 0].legend(title="Musim")
        
        # Line Chart 2: Total Sewa Sepeda untuk Setiap Musim
        sns.lineplot(x="season_name", y="cnt", data=day_df, estimator="sum", ci=None, marker="o", color="purple", ax=axes[1, 1])
        axes[1, 1].set_title("Total Sewa Sepeda untuk Setiap Musim", fontsize=14)
        axes[1, 1].set_ylabel("Jumlah Sewa (Juta)", fontsize=12)
        axes[1, 1].set_xlabel("Musim")
        axes[1, 1].yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}")

        plt.tight_layout()

        # Tampilkan visualisasi di Streamlit
        st.pyplot(fig)
        
        # Tambahkan insight
        st.markdown("""
        ### **Insight:**
        - **Total Sewa Sepeda Berdasarkan Musim (Bar Chart - Kiri Atas):**
            - Musim Fall (Gugur) memiliki total penyewaan tertinggi (~1,06 juta), disusul oleh Summer (Musim Panas) dan Winter (Musim Dingin), mengindikasikan bahwa musim ini memiliki kondisi optimal bagi pengguna untuk bersepeda, mungkin karena suhu yang nyaman dan kondisi cuaca yang mendukung.
            - Spring (Musim Semi) memiliki jumlah sewa terendah (~471 ribu), sekitar setengah dari Fall, mengindikasikan bahwa faktor cuaca atau tingkat aktivitas pengguna yang lebih rendah di musim ini bisa menjadi penyebabnya.
        - **Rata-Rata Sewa Sepeda Harian Berdasarkan Musim (Bar Chart - Kanan Atas):**
            - Rata-rata sewa harian juga tertinggi pada Fall (5.644 sewa/hari), menunjukkan tingginya permintaan saat musim ini, mengindikasikan bahwa periode ini merupakan waktu yang sangat produktif untuk bisnis penyewaan sepeda. 
            - Spring memiliki rata-rata sewa harian terendah (~2.604 sewa/hari), mengindikasikan faktor cuaca atau preferensi pengguna yang menyebabkan minat bersepeda lebih rendah. 
            - Summer dan Winter memiliki angka sewa harian yang cukup seimbang, mengindikasikan bahwa meskipun ada perbedaan suhu ekstrem di kedua musim ini, masih terdapat minat yang cukup tinggi dalam penyewaan sepeda.
        - **Tren Sewa Sepeda Berdasarkan Musim (Line Chart - Kiri Bawah):**
            - Tren menunjukkan peningkatan sewa secara bertahap dari awal tahun hingga mencapai puncaknya pada pertengahan tahun, kemudian menurun menjelang akhir tahun, mengindikasikan adanya pola musiman dalam penyewaan sepeda. 
            - Fluktuasi harian cukup tinggi, terutama di musim panas dan gugur, yang mungkin disebabkan oleh variasi cuaca atau aktivitas pengguna, mengindikasikan bahwa faktor lingkungan dan gaya hidup mempengaruhi tingkat penyewaan secara signifikan.
        - **Total Sewa Sepeda untuk Setiap Musim (Line Chart - Kanan Bawah):**
            - Visualisasi ini mengonfirmasi bahwa jumlah sewa meningkat dari Spring → Summer → Fall lalu menurun saat memasuki Winter mengindikasikan adanya siklus tahunan yang dapat digunakan untuk strategi bisnis.
            - Fall menjadi musim paling optimal untuk penyewaan, sementara Spring memiliki permintaan paling rendah, mengindikasikan bahwa bisnis dapat memanfaatkan tren ini untuk menyesuaikan strategi operasional.<br>

        Musim Fall merupakan musim paling populer untuk penyewaan sepeda, baik dari total maupun rata-rata harian.
        Spring memiliki jumlah penyewaan terendah, mungkin karena kondisi cuaca atau kurangnya minat masyarakat untuk bersepeda di periode ini.
        Winter masih memiliki angka sewa yang cukup tinggi, kemungkinan karena pengguna yang sudah terbiasa menggunakan sepeda dalam kondisi dingin.
        Pola tren musiman ini bisa digunakan untuk strategi bisnis, seperti meningkatkan jumlah sepeda di musim Fall atau menawarkan promosi saat Spring untuk meningkatkan pemakaian.  
        """)

    # Pertanyaan 2
    elif question == "Bagaimana pengaruh kondisi cuaca terhadap pola penyewaan sepeda?":
        weather_stats = day_df.groupby("weather_condition")["cnt"].agg(["mean", "sum"]).sort_values("sum", ascending=False)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        sns.barplot(x=weather_stats.index, y=weather_stats["sum"], palette="viridis", ax=axes[0, 0])
        axes[0, 0].set_title("Total Sewa Sepeda Berdasarkan Kondisi Cuaca")

        sns.barplot(x=weather_stats.index, y=weather_stats["mean"], palette="viridis", ax=axes[0, 1])
        axes[0, 1].set_title("Rata-rata Sewa Sepeda Harian Berdasarkan Kondisi Cuaca")

        sns.lineplot(data=day_df, x="dteday", y="cnt", hue="weather_condition", palette="viridis", linewidth=2, ax=axes[1, 0])
        axes[1, 0].set_title("Tren Sewa Sepeda Berdasarkan Kondisi Cuaca")

        sns.lineplot(x="weather_condition", y="cnt", data=day_df, estimator="sum", ci=None, marker="o", color="purple", ax=axes[1, 1])
        axes[1, 1].set_title("Total Sewa Sepeda untuk Setiap Kondisi Cuaca")

        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### **Insight:**
        - **Total Sewa Sepeda Berdasarkan Kondisi Cuaca (Bar Chart - Kiri Atas):**
            - Penyewaan sepeda paling tinggi terjadi pada kondisi Clear/Partly Cloudy (~2,26 juta sewa). 
            - Penyewaan menurun pada kondisi Mist/Cloudy (~996 ribu sewa). 
            - Light Precipitation (hujan ringan) memiliki jumlah sewa yang sangat rendah (~37 ribu), menunjukkan bahwa hujan sangat menghambat penggunaan sepeda.
        - **Rata-Rata Sewa Sepeda Harian Berdasarkan Kondisi Cuaca (Bar Chart - Kanan Atas):**
            - Rata-rata sewa tertinggi terjadi saat Clear/Partly Cloudy (4.877 sewa/hari). 
            - Mist/Cloudy masih memiliki angka yang cukup tinggi (4.036 sewa/hari), meskipun lebih rendah dari kondisi cerah. 
            - Light Precipitation memiliki angka yang sangat rendah (1.803 sewa/hari), mengindikasikan dampak negatif hujan terhadap minat pengguna.
        - **Tren Sewa Sepeda Berdasarkan Kondisi Cuaca (Line Chart - Kiri Bawah):**
            - Sewa sepeda meningkat seiring waktu, terutama pada hari cerah dan berawan. 
            - Penurunan signifikan terlihat saat terjadi hujan ringan, menunjukkan bahwa pengguna menghindari bersepeda dalam kondisi ini. 
            - Tren harian menunjukkan fluktuasi besar, kemungkinan dipengaruhi oleh faktor eksternal seperti suhu dan hari kerja vs. akhir pekan.
        - **Total Sewa Sepeda untuk Setiap Kondisi Cuaca (Line Chart - Kanan Bawah):**
            - Grafik ini mempertegas bahwa Clear/Partly Cloudy adalah kondisi terbaik untuk penyewaan sepeda. 
            - Mist/Cloudy masih memiliki pangsa pasar yang besar dan bisa dioptimalkan. 
            - Light Precipitation memiliki jumlah penyewaan yang sangat kecil, menunjukkan perlunya strategi alternatif di kondisi ini.

        Pengguna lebih cenderung menyewa sepeda saat cuaca cerah atau sedikit berawan.
        Cuaca berkabut masih memungkinkan penggunaan sepeda, tetapi dengan penurunan permintaan yang cukup besar. Saat hujan ringan, permintaan turun drastis, mengindikasikan bahwa pengguna lebih memilih alternatif transportasi lain atau menghindari aktivitas bersepeda.
        """)

    # Pertanyaan 3
    elif question == "Bagaimana tren penyewaan sepeda per jam sepanjang hari, dan kapan waktu penggunaan tertinggi?":
        
        # Urutan hari dalam seminggu
        day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        # Buat visualisasi dalam 1 baris 2 kolom
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Line Chart - Tren Sewa per Jam
        sns.lineplot(x="hr", y="cnt", data=hour_df, estimator="sum", ci=None, marker="o", color="purple", ax=axes[0])
        axes[0].set_title("Tren Sewa Sepeda per Jam dalam Sehari", fontsize=14)
        axes[0].set_ylabel("Jumlah Sewa (Ribu)", fontsize=12)
        axes[0].set_xlabel("Jam", fontsize=12)
        axes[0].set_xticks(range(0, 24, 1))  # Set grid berdasarkan jam (0-24)
        axes[0].grid(axis="both", linestyle="--", alpha=0.6)  # Tambahkan grid
        axes[0].yaxis.set_major_formatter(lambda x, _: f"{x/1e3:.0f}")

        # Heatmap - Sewa Sepeda per Jam dan Hari
        heatmap_data = hour_df.pivot_table(values="cnt", index="day_of_week", columns="hr", aggfunc="sum")
        heatmap_data = heatmap_data.reindex(day_order)  # Urutkan berdasarkan hari yang benar

        sns.heatmap(heatmap_data, cmap="viridis", ax=axes[1])
        axes[1].set_title("Pola Sewa Sepeda (Jam vs. Hari)", fontsize=14)
        axes[1].set_ylabel("Hari", fontsize=12)
        axes[1].set_xlabel("Jam", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### **Insight:**
        - **Tren Sewa Sepeda per Jam dalam Sehari:** 
            - Jumlah penyewaan sepeda rendah pada dini hari (00:00 - 05:00), dengan titik terendah sekitar pukul 04:00. 
            - Lonjakan signifikan terjadi sekitar pukul 07:00 - 09:00, dengan puncak pertama sekitar pukul 08:00 (~160 ribu sewa). 
            - Setelah itu, jumlah penyewaan menurun hingga siang hari, tetapi kembali meningkat pada sore hari. 
            - Puncak penyewaan tertinggi terjadi sekitar pukul 17:00 - 19:00, dengan titik maksimal pada pukul 18:00 (~210 ribu sewa). 
            - Setelah pukul 19:00, jumlah penyewaan menurun secara bertahap hingga malam hari.
            - Lonjakan pagi menunjukkan bahwa sepeda digunakan sebagai alat transportasi untuk perjalanan ke kantor/sekolah.
            - Lonjakan sore-malam menunjukkan penggunaan sepeda untuk perjalanan pulang kerja/sekolah serta aktivitas rekreasi atau olahraga.
            - Tren ini mengindikasikan bahwa mayoritas pengguna adalah pekerja atau pelajar yang menggunakan sepeda sebagai transportasi utama dalam jam sibuk.

        - **Pola Sewa Sepeda (Jam vs. Hari):**
            - Senin - Jumat: Pola penyewaan menunjukkan dua puncak utama pada pagi (~08:00) dan sore (~19:00), mencerminkan jam sibuk. 
            - Sabtu - Minggu: Tren berbeda, di mana penyewaan meningkat lebih lambat di pagi hari dan puncak lebih merata di siang hingga sore (~10:00 - 18:00).
            - Warna terang pada heatmap menunjukkan intensitas penyewaan tertinggi, yang terutama terjadi pada sore hari di hari kerja. 
            - Hari kerja memiliki pola sewa yang lebih terstruktur karena keterikatan jadwal kerja dan sekolah. 
            - Akhir pekan menunjukkan pola yang lebih fleksibel, dengan sewa meningkat secara bertahap dan tersebar sepanjang hari. 
            - Ini mengindikasikan adanya perbedaan tujuan penggunaan sepeda: transportasi pada hari kerja dan rekreasi pada akhir pekan.
        """)
        

    #  Pertanyaan 4
    elif question == "Apakah terdapat perbedaan signifikan dalam pola penyewaan sepeda antara hari kerja dan akhir pekan?":
        # Buat visualisasi dalam 1 baris 2 kolom
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Bar Chart Grouped - Total Sewa di Hari Kerja vs. Akhir Pekan
        sns.barplot(x="workingday", y="cnt", data=day_df, estimator=sum, ci=None, palette="viridis", ax=axes[0])
        axes[0].set_title("Total Sewa Sepeda: Hari Kerja vs. Akhir Pekan", fontsize=14)
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(["Akhir Pekan", "Hari Kerja"], fontsize=12)
        axes[0].set_ylabel("Jumlah Sewa (Juta)", fontsize=12)
        axes[0].set_xlabel("Tipe Hari", fontsize=12)
        axes[0].yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}")

        # Tambahkan angka di atas masing-masing bin
        for i, v in enumerate(day_df.groupby("workingday")["cnt"].sum()):
            axes[0].text(i, v + 3000, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")

        # Line Chart - Tren Sewa di Hari Kerja vs. Akhir Pekan
        sns.lineplot(x="hr", y="cnt", hue="workingday", data=hour_df, estimator=sum, ci=None, marker="o", palette=["blue", "green"], ax=axes[1])
        axes[1].set_title("Tren Sewa Sepeda Sepanjang Hari: Hari Kerja vs. Akhir Pekan", fontsize=14)
        axes[1].set_ylabel("Jumlah Sewa (Ribu)", fontsize=12)
        axes[1].set_xlabel("Jam", fontsize=12)
        axes[1].set_xticks(range(0, 24, 1))
        axes[1].grid(axis="both", linestyle="--", alpha=0.6)
        axes[1].legend(["Akhir Pekan", "Hari Kerja"], title="Kategori", fontsize=11)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}"))

        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### **Insight:**
        **Insight:**
        - **Total Penyewaan Sepeda: Hari Kerja vs. Akhir Pekan (Bar Chart - Kiri):**
            - Hari Kerja memiliki total penyewaan sepeda sebanyak 2,292,410 unit. 
            - Akhir Pekan memiliki total penyewaan sepeda sebanyak 1,000,269 unit. 
            - Jumlah penyewaan sepeda pada hari kerja sekitar 2,3 kali lipat dibandingkan akhir pekan.
            - Sepeda lebih sering digunakan sebagai alat transportasi utama pada hari kerja, kemungkinan besar untuk keperluan perjalanan ke kantor/sekolah.
            - Pada akhir pekan, penggunaan sepeda lebih rendah, mengindikasikan bahwa penggunaannya lebih bersifat rekreasi atau santai dibandingkan kebutuhan transportasi sehari-hari.
        - **Tren Penyewaan Sepeda Sepanjang Hari: Hari Kerja vs. Akhir Pekan (Line Chart - Kanan):**
            - Hari Kerja: Pola penyewaan memiliki dua puncak utama: Pukul 08:00 (sekitar 145 ribu sewa) → Perjalanan ke kantor/sekolah. Pukul 18:00 - 19:00 (sekitar 160 ribu sewa) → Perjalanan pulang kerja/sekolah. Setelah pukul 19:00, jumlah penyewaan turun drastis.
            - Akhir Pekan: Pola lebih stabil tanpa lonjakan ekstrem. Penyewaan mulai meningkat dari pagi hari dan mencapai puncaknya antara 10:00 - 16:00, dengan sekitar 70 - 80 ribu sewa per jam. Jumlah sewa tetap lebih tinggi dibandingkan dini hari/malam, tetapi tidak ada lonjakan signifikan seperti di hari kerja
            - Hari kerja menunjukkan pola yang lebih tajam dengan lonjakan di pagi dan sore hari, mengindikasikan bahwa sepeda lebih sering digunakan sebagai alat transportasi utama pada hari kerja, kemungkinan besar untuk keperluan perjalanan ke kantor/sekolah.
            - Akhir pekan memiliki pola yang lebih merata, menunjukkan penggunaan sepeda untuk rekreasi, olahraga, atau aktivitas santai, bukan sebagai transportasi utama.
            - Tingkat penyewaan lebih tinggi di sore hari pada hari kerja dibandingkan akhir pekan, yang bisa dikaitkan dengan kepadatan lalu lintas dan kebutuhan perjalanan pulang.
        """)

    # Pertanyaan 5
    elif question == "Bagaimana distribusi dan rasio antara pengguna kasual dan terdaftar di berbagai periode waktu?":
        sfig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Bar Chart Total Pengguna Kasual vs. Terdaftar
        user_distribution = day_df[["casual", "registered"]].sum()
        sns.barplot(x=user_distribution.index, y=user_distribution.values, palette="viridis", ax=axes[0])
        axes[0].set_title("Total Pengguna Kasual vs. Terdaftar", fontsize=14)
        axes[0].set_xlabel("Tipe Pengguna", fontsize=12)
        axes[0].set_ylabel("Jumlah Sewa (Juta)", fontsize=12)
        axes[0].yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}")

        # Tambahkan angka di atas masing-masing bin
        for i, v in enumerate(user_distribution.values):
            axes[0].text(i, v + 1500, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")

        # Bar Chart Stacked - Pengguna Kasual vs. Terdaftar per Jam
        hourly_grouped = hour_df.groupby("hr")[["casual", "registered"]].sum().reset_index()
        hourly_grouped.plot(x="hr", kind="bar", stacked=True, colormap="viridis", ax=axes[1])
        axes[1].set_title("Perbandingan Pengguna Kasual dan Terdaftar per Jam", fontsize=14)
        axes[1].set_xlabel("Jam", fontsize=12)
        axes[1].set_ylabel("Jumlah Sewa", fontsize=12)
        axes[1].legend(["Casual", "Registered"])

        # Pie Chart - Rasio Pengguna Kasual vs. Terdaftar
        axes[2].pie(user_distribution, labels=["Casual", "Registered"], autopct="%1.1f%%", colors=sns.color_palette("viridis", 2), startangle=90)
        axes[2].set_title("Rasio Pengguna Kasual vs. Terdaftar", fontsize=14)

        plt.tight_layout()
        st.pyplot(sfig)
        
        st.markdown("""
        ### **Insight:**
        **Insight:**
        - **Total Pengguna Kasual vs. Terdafar (Bar Chart - Kiri):**
            - Registered users mendominasi dengan total 2,672,662 penyewaan sepeda. 
            - Casual users memiliki total penyewaan jauh lebih rendah, hanya 620,017. 
            - Registered users berkontribusi lebih dari 4 kali lipat dibanding casual users dalam hal jumlah penyewaan.
            - Sebagian besar penyewaan dilakukan oleh pengguna terdaftar, yang kemungkinan besar menggunakan sepeda untuk keperluan transportasi harian (misalnya perjalanan kerja atau sekolah).
            - Casual users lebih sedikit karena mungkin mereka hanya menyewa untuk rekreasi atau kebutuhan sesekali.
        - **Perbandingan Pengguna Kasual dan Terdaftar per Jam (Bar Chart - Tengah):**
            - Registered users memiliki pola penyewaan yang kuat pada jam sibuk, terutama pukul 07:00 - 09:00 dan 17:00 - 19:00. 
            - Casual users lebih dominan di siang hari, dengan lonjakan bertahap dari 10:00 - 16:00. 
            - Saat jam sibuk, jumlah penyewaan registered users jauh lebih tinggi dibanding casual users
            - Registered users lebih banyak beraktivitas pada hari kerja, yang sesuai dengan pola perjalanan pekerja kantoran atau pelajar.
            - Casual users lebih aktif pada siang hari, kemungkinan besar karena mereka menggunakan sepeda untuk wisata, jalan santai, atau aktivitas rekreasi.
            - Lonjakan penyewaan registered users pada pagi dan sore hari menunjukkan pola perjalanan kerja pulang-pergi, sedangkan penyewaan casual users lebih merata tanpa lonjakan ekstrem.
        - **Rasio Pengguna Kasual vs. Terdaftar (Pie Chart - Kanan):**
            - Registered users mendominasi dengan 81.2% dari total penyewaan, sedangkan casual users hanya 18.8%.
            - Sebagian besar pelanggan adalah pengguna setia yang berlangganan layanan penyewaan sepeda, sehingga strategi bisnis bisa lebih fokus pada mempertahankan dan meningkatkan layanan bagi mereka.
            - Casual users memiliki porsi kecil, sehingga ada peluang untuk meningkatkan pangsa pasar dengan menawarkan promosi atau paket fleksibel bagi pelanggan non-terdaftar.
        """)
    
    # Pertanyaan 6
    elif question == "Bagaimana dampak hari libur terhadap pola penyewaan sepeda dibandingkan dengan hari biasa?":
        # Buat visualisasi dalam 1 baris 2 kolom
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Bar Chart Grouped - Total Sewa di Hari Libur vs. Hari Biasa
        sns.barplot(x="holiday", y="cnt", data=day_df, estimator=sum, ci=None, palette="viridis", ax=axes[0])
        axes[0].set_title("Total Sewa Sepeda: Hari Libur vs. Hari Biasa", fontsize=14)
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(["Hari Biasa", "Hari Libur"], fontsize=12)
        axes[0].set_ylabel("Jumlah Sewa (Juta)", fontsize=12)
        axes[0].set_xlabel("Kategori Hari", fontsize=12)

        # Format y-label ke dalam ribuan untuk keterbacaan
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))

        # Tambahkan angka di atas masing-masing bin
        for i, v in enumerate(day_df.groupby("holiday")["cnt"].sum()):
            axes[0].text(i, v + 3000, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")

        # Line Chart - Tren Sewa di Hari Libur vs. Hari Biasa
        sns.lineplot(x="hr", y="cnt", hue="holiday", data=hour_df, estimator=sum, ci=None, marker="o", palette=["blue", "green"], ax=axes[1])
        axes[1].set_title("Tren Sewa Sepeda Sepanjang Hari: Hari Libur vs. Hari Biasa", fontsize=14)
        axes[1].set_ylabel("Jumlah Sewa (Ribu)", fontsize=12)
        axes[1].set_xlabel("Jam", fontsize=12)
        axes[1].set_xticks(range(0, 24, 1))  # Ubah grid ke 24 jam
        axes[1].grid(axis="both", linestyle="--", alpha=0.6)  # Tambahkan grid
        axes[1].legend(["Hari Biasa", "Hari Libur"], title="Kategori", fontsize=11)

        # Format y-label menjadi ribuan
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}"))

        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### **Insight:**
        **Insight:**
        - **Total Sewa Sepeda: Hari Libur vs. Hari Biasa (Bar Chart - Kiri):**
            - Penyewaan sepeda pada hari biasa (regular) jauh lebih tinggi dibandingkan dengan hari libur (holiday).
            - Selisih jumlah penyewaan yang signifikan menunjukkan bahwa sepeda lebih sering digunakan pada hari-hari kerja atau aktivitas rutin dibandingkan saat libur.
        - **Tren Sewa Sepeda Sepanjang Hari: Hari Libur vs. Hari Biasa (Line Chart - Kanan):**
            - Hari biasa memiliki pola penyewaan yang teratur dengan lonjakan signifikan pada pagi (07:00 - 09:00) dan sore (17:00 - 19:00), yang menandakan pola perjalanan kerja.
            - Hari libur menunjukkan pola yang lebih merata, dengan peningkatan bertahap dari pagi hingga sore, tanpa lonjakan signifikan.
        """)

    # Pertanyaan 7
    elif question == "Apakah terdapat korelasi antara suhu, kelembaban, kecepatan angin, dan jumlah penyewaan sepeda?":
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Heatmap - Correlation Matrix
        corr_matrix = day_df[["cnt", "temp", "hum", "windspeed"]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="viridis", ax=axes[0])
        axes[0].set_title("Korelasi antara Faktor Cuaca dan Jumlah Penyewaan")

        # Scatter Plot dengan Correlation Line - Temp vs. Penyewaan
        sns.regplot(x="temp", y="cnt", data=day_df, ax=axes[1], color="purple")
        axes[1].set_title("Hubungan antara Suhu dan Jumlah Penyewaan")
        axes[1].set_xlabel("Suhu")
        axes[1].set_ylabel("Jumlah Penyewaan")

        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### **Insight:**
        **Insight:**
        - **Korelasi antara Faktor Cuaca dan Jumlah Penyewaan (Heatmap - Kiri):**
            - Suhu (temp) memiliki korelasi positif yang cukup kuat (0.63) terhadap jumlah penyewaan sepeda, yang berarti semakin tinggi suhu, semakin banyak sepeda yang disewa.
            - Kelembaban (hum) memiliki korelasi negatif lemah (-0.1), menunjukkan bahwa kelembaban tidak terlalu mempengaruhi jumlah penyewaan.
            - Kecepatan angin (windspeed) memiliki korelasi negatif (-0.23), yang berarti semakin kencang angin, semakin sedikit jumlah penyewaan sepeda, meskipun pengaruhnya tidak terlalu besar.
            - Suhu merupakan faktor lingkungan yang paling mempengaruhi penyewaan sepeda, karena cuaca yang lebih hangat cenderung lebih nyaman untuk bersepeda.
            - Kelembaban tidak terlalu signifikan, mungkin karena pengguna lebih memperhatikan suhu dibandingkan tingkat kelembaban udara.
            - Kecepatan angin yang tinggi bisa membuat perjalanan bersepeda menjadi lebih sulit, sehingga dapat mengurangi jumlah penyewaan sepeda.
        - **Hubungan antara Suhu dan Jumlah Penyewaan (Scatter Plot - Kanan):**
            - Scatter plot menunjukkan tren positif yang jelas antara suhu dan jumlah penyewaan, dikonfirmasi dengan garis regresi yang menunjukkan peningkatan jumlah sewa saat suhu naik.
            - Meskipun ada beberapa penyebaran data yang variatif, tren keseluruhannya tetap menunjukkan hubungan positif yang kuat.
            - Suhu yang lebih tinggi mendorong lebih banyak orang untuk menyewa sepeda, kemungkinan karena cuaca lebih nyaman untuk bersepeda.
            - Namun, pada suhu yang sangat tinggi (di atas titik tertentu), mungkin ada titik jenuh di mana penyewaan mulai menurun karena cuaca menjadi terlalu panas.
        """)
    
if selected == "Clustering":
    st.title("Clustering")

    def categorize_time(hour):
        if 5 <= hour <= 11:
            return "Morning"
        elif 12 <= hour <= 16:
            return "Afternoon"
        elif 17 <= hour <= 20:
            return "Evening"
        else:
            return "Night"

    # Tambahkan kolom kategori waktu
    hour_df["TimePeriod"] = hour_df["hr"].apply(categorize_time)

    # Kelompokkan data berdasarkan TimePeriod dan holiday
    time_period_clusters = hour_df.groupby(["TimePeriod", "holiday"]).agg({
        "casual": "mean",
        "registered": "mean"
    }).reset_index()

    # Generate warna dari colormap Viridis
    viridis = cm.get_cmap("viridis", 4)
    colors = {
        "holiday_casual": mcolors.to_hex(viridis(0)),
        "non_holiday_casual": mcolors.to_hex(viridis(1)),
        "holiday_registered": mcolors.to_hex(viridis(2)),
        "non_holiday_registered": mcolors.to_hex(viridis(3))
    }

    # Visualisasi hasil clustering
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35  # Lebar bar
    time_labels = ["Morning", "Afternoon", "Evening", "Night"]
    index = np.arange(len(time_labels))

    # Pisahkan data berdasarkan kategori hari libur dan non-libur
    holiday_data = time_period_clusters[time_period_clusters["holiday"] == 1]
    non_holiday_data = time_period_clusters[time_period_clusters["holiday"] == 0]

    # Plot stacked bars untuk casual users
    ax.bar(index, 
        holiday_data["casual"], 
        bar_width, label="Casual (Holiday)", color=colors["holiday_casual"])

    ax.bar(index, 
        non_holiday_data["casual"], 
        bar_width, bottom=holiday_data["casual"], label="Casual (Non-Holiday)", color=colors["non_holiday_casual"])

    # Plot stacked bars untuk registered users
    ax.bar(index + bar_width, 
        holiday_data["registered"], 
        bar_width, label="Registered (Holiday)", color=colors["holiday_registered"])

    ax.bar(index + bar_width, 
        non_holiday_data["registered"], 
        bar_width, bottom=holiday_data["registered"], label="Registered (Non-Holiday)", color=colors["non_holiday_registered"])

    # Tambahkan label dan legenda
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Average Rentals")
    ax.set_title("Average Rentals by Time Period (Casual vs Registered)")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(time_labels)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    ### **Insight:**
    #### **Pola Penyewaan Berdasarkan Waktu dan Jenis Pengguna:**
    - **Pagi hari (Morning):** Pengguna terdaftar (registered) mendominasi penyewaan sepeda, terutama pada hari kerja. Pengguna kasual lebih sedikit.
    - **Siang hingga sore (Afternoon):** Penyewaan meningkat drastis, terutama oleh pengguna terdaftar.
    - **Malam hari (Evening & Night):** Penyewaan berkurang secara signifikan, terutama dari pengguna kasual.
    - **Pengguna terdaftar (registered)** lebih sering menyewa di pagi dan sore hari, yang menunjukkan pola perjalanan komuter *(work-home travel)*.
    - **Pengguna kasual** cenderung menyewa di siang dan sore hari, yang kemungkinan besar terkait dengan aktivitas rekreasi atau wisata.
    - **Hari libur** memiliki lebih banyak penyewaan oleh pengguna kasual dibandingkan hari kerja.

    #### **Perbedaan antara Hari Libur dan Hari Kerja:**
    - **Hari kerja:** Pengguna terdaftar mendominasi, terutama di pagi dan sore hari.
    - **Hari libur:** Pengguna kasual meningkat drastis di siang dan sore hari, tetapi masih lebih sedikit dibandingkan pengguna terdaftar secara keseluruhan.
    - **Layanan penyewaan** dapat meningkatkan kapasitas sepeda di pagi dan sore hari untuk mengakomodasi pengguna terdaftar.
    - **Strategi pemasaran** bisa difokuskan pada pengguna kasual di hari libur, misalnya dengan paket diskon atau rute wisata menarik.
    - **Jam operasional** bisa disesuaikan agar lebih fleksibel di akhir pekan untuk meningkatkan jumlah penyewaan malam.

    #### **Potensi Optimalisasi Bisnis Berdasarkan Pola Penyewaan:**
    - **Optimalisasi jumlah sepeda** → Tambah kapasitas sepeda di pagi dan sore hari untuk mengakomodasi pengguna terdaftar.
    - **Strategi harga dinamis** → Tarif lebih tinggi pada jam sibuk *(morning & evening)* dan promo diskon untuk pengguna kasual di siang hari.
    - **Fokus layanan di hari libur** → Promosi dan event khusus bagi pengguna kasual di siang dan sore hari.
    - **Pemanfaatan data untuk prediksi permintaan** → Menyesuaikan jumlah sepeda berdasarkan prediksi pola sewa di setiap waktu.
    """)

if selected == "Conclusion":
    st.title("Conclusion & Recommendation")
    
    st.subheader("Conclusion")
    st.markdown("""
    - **Pola Musiman:** Penyewaan sepeda tertinggi terjadi di musim gugur dan terendah di musim dingin.
    - **Pengaruh Cuaca:** Cuaca buruk (hujan/salju) menurunkan jumlah penyewaan, terutama bagi pengguna kasual.
    - **Tren Harian:** Hari kerja didominasi oleh pengguna terdaftar, sedangkan akhir pekan lebih banyak menarik pengguna kasual.
    - **Tren Per Jam:**
        - Puncak penyewaan terjadi pada pagi (07:00-09:00) dan sore (17:00-19:00) di hari kerja.
        - Akhir pekan memiliki pola penyewaan yang lebih merata sepanjang siang hingga sore.
    - **Perbedaan Pengguna:**
        - Pengguna terdaftar lebih konsisten menyewa sepanjang tahun.
        - Pengguna kasual sangat dipengaruhi oleh musim dan cuaca.
    - **Korelasi Faktor Lingkungan:**
        - Suhu lebih tinggi meningkatkan penyewaan.
        - Kelembaban dan kecepatan angin yang tinggi menurunkan penyewaan.
    - **Clustering Berdasarkan Waktu:**
        - **Pagi & Sore:** Dominasi pengguna terdaftar (perjalanan komuter).
        - **Siang:** Dominasi pengguna kasual (aktivitas rekreasi).
        - **Malam:** Penyewaan rendah untuk semua kategori pengguna.
    """)
    
    st.subheader("Recommendations")
    st.markdown("""
    - **Penyesuaian Sesuai Musim:**
        - Tambah sepeda di musim gugur (puncak penyewaan).
        - Kurangi operasional di musim dingin atau tawarkan promosi untuk menarik pengguna kasual.
    - **Strategi Cuaca & Diskon Dinamis:**
        - Tawarkan diskon atau insentif bagi pengguna kasual saat cuaca kurang bersahabat.
        - Sediakan stasiun penampungan atau shelter sepeda di lokasi strategis saat cuaca buruk.
    - **Strategi Layanan Berdasarkan Hari:**
        - Tingkatkan promosi paket wisata atau sewa harian untuk pengguna kasual di akhir pekan.
    - **Penyesuaian Tarif Sesuai Waktu:**
        - Tarif premium saat jam sibuk (pagi & sore hari kerja).
        - Diskon untuk siang hari guna menarik lebih banyak pengguna kasual.
    - **Inovasi & Promosi untuk Malam Hari:**
        - Kampanye atau event khusus seperti “Night Ride” untuk meningkatkan penyewaan malam hari.
        - Peningkatan keamanan dan penerangan di jalur sepeda untuk mendorong penggunaan malam.
    - **Kerjasama dengan Bisnis Lokal:**
        - Kolaborasi dengan tempat wisata, restoran, dan hotel untuk menyediakan paket sewa sepeda dengan diskon.
        - Sponsor atau iklan di sepeda untuk menambah sumber pendapatan.
    """)