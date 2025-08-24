import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# Page configuration
st.set_page_config(
    page_title="Deteksi Penyakit Mata Fundus",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        color: #000;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #fff3cd;
        color: #000;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['Katarak', 'Retinopati Diabetik', 'Glaukoma', 'Normal']
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Functions
@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        # Method 1: Try loading with compile=False
        model_path = "./best_fundus_model.h5"
        if os.path.exists(model_path):
            st.info(f"Memuat model dari: {os.path.abspath(model_path)}")
            
            try:
                # First try: Load without compilation
                model = tf.keras.models.load_model(model_path, compile=False)
                st.success("Model berhasil dimuat (tanpa kompilasi)")
                return model
            except Exception as e1:
                st.warning(f"Gagal memuat tanpa kompilasi: {e1}")
                
                try:
                    # Second try: Load with custom objects
                    model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects=None,
                        compile=False
                    )
                    st.success("Model berhasil dimuat dengan objek kustom")
                    return model
                except Exception as e2:
                    st.error(f"Gagal dengan objek kustom: {e2}")
                    
                    try:
                        # Third try: Load with different method
                        model = tf.keras.models.load_model(model_path)
                        st.success("Model berhasil dimuat dengan metode default")
                        return model
                    except Exception as e3:
                        st.error(f"Semua metode pemuatan gagal: {e3}")
                        return None
        else:
            st.error("File model tidak ditemukan")
            return None
            
    except Exception as e:
        st.error(f"Kesalahan tak terduga: {str(e)}")
        st.write(f"Jenis kesalahan: {type(e).__name__}")
        return None

def preprocess_image(image, img_size=224):
    """Preprocess image for prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize image
    img_resized = cv2.resize(img_array, (img_size, img_size))
    
    # Normalize pixel values
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_disease(model, image, class_names):
    """Make prediction on the image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class name
        predicted_class = class_names[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        
        return predicted_class, confidence, class_probabilities
    
    except Exception as e:
        st.error(f"Kesalahan dalam prediksi: {str(e)}")
        return None, None, None

def create_probability_chart(probabilities):
    """Create a bar chart of class probabilities"""
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig = px.bar(
        x=probs, 
        y=classes, 
        orientation='h',
        title="Probabilitas Kelas",
        labels={'x': 'Probabilitas', 'y': 'Kelas Penyakit'},
        color=probs,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def get_disease_info(disease_name):
    """Get information about the disease"""
    disease_info = {
        'Katarak': {
            'description': 'Kekeruhan pada lensa mata yang mempengaruhi penglihatan. Sebagian besar katarak berkaitan dengan penuaan.',
            'symptoms': ['Penglihatan keruh atau buram', 'Warna terlihat pudar', 'Silau', 'Penglihatan malam yang buruk'],
            'treatment': 'Operasi adalah pengobatan yang paling efektif untuk katarak.',
            'prevention': 'Lindungi mata dari sinar UV, berhenti merokok, pertahankan diet sehat'
        },
        'Retinopati Diabetik': {
            'description': 'Komplikasi diabetes yang mempengaruhi mata yang disebabkan oleh kerusakan pembuluh darah retina.',
            'symptoms': ['Bintik-bintik atau garis gelap mengambang dalam penglihatan', 'Penglihatan buram', 'Penglihatan berfluktuasi', 'Area gelap dalam penglihatan'],
            'treatment': 'Kontrol gula darah, pengobatan laser, vitrektomi, injeksi obat',
            'prevention': 'Kelola diabetes, pemeriksaan mata rutin, pertahankan tekanan darah yang sehat'
        },
        'Glaukoma': {
            'description': 'Kelompok kondisi mata yang merusak saraf optik, sering disebabkan oleh tekanan mata yang abnormal tinggi.',
            'symptoms': ['Kehilangan penglihatan tepi secara bertahap', 'Penglihatan terowongan', 'Nyeri mata', 'Mual dan muntah'],
            'treatment': 'Tetes mata, pengobatan laser, operasi untuk menurunkan tekanan mata',
            'prevention': 'Pemeriksaan mata rutin, olahraga teratur, batasi asupan kafein'
        },
        'Normal': {
            'description': 'Mata yang sehat tanpa kelainan yang terdeteksi dalam gambar fundus.',
            'symptoms': ['Penglihatan jernih', 'Tidak ada gangguan visual', 'Penampilan retina yang sehat'],
            'treatment': 'Lanjutkan pemeriksaan mata rutin untuk menjaga kesehatan mata',
            'prevention': 'Pertahankan gaya hidup sehat, lindungi mata dari UV, pemeriksaan mata rutin'
        }
    }
    return disease_info.get(disease_name, {})

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Sistem Deteksi Penyakit Mata Fundus</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih halaman", 
                               ["Beranda", "Deteksi Penyakit", "Informasi Model", "Statistik Dataset", "Tentang"])
    
    if page == "Beranda":
        home_page()
    elif page == "Deteksi Penyakit":
        detection_page()
    elif page == "Informasi Model":
        model_info_page()
    elif page == "Statistik Dataset":
        dataset_stats_page()
    elif page == "Tentang":
        about_page()

def home_page():
    """Home page content"""
    st.markdown('<h2 class="subheader">Selamat Datang di Sistem Deteksi Penyakit Mata Fundus</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Gambaran Sistem
        Sistem bertenaga AI ini menggunakan pembelajaran mendalam untuk menganalisis gambar fundus dan mendeteksi berbagai penyakit mata termasuk:
        - **Katarak**: Kekeruhan pada lensa mata
        - **Retinopati Diabetik**: Kerusakan pembuluh darah retina akibat diabetes
        - **Glaukoma**: Kerusakan saraf optik
        - **Normal**: Kondisi mata yang sehat
        
        ### 🔬 Cara Kerja
        1. **Unggah** gambar fundus
        2. **Analisis AI** menggunakan model CNN terlatih
        3. **Hasil** dengan skor kepercayaan dan rekomendasi
        
        ### 📊 Kinerja Model
        Sistem kami mencapai akurasi tinggi dalam mendeteksi penyakit mata dengan berbagai arsitektur CNN termasuk model kustom dan pendekatan transfer learning.
        """)
    
    with col2:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTclmqHhG8Nro8Bk8PfjbrvONKnqqeuL07bacdaCSIIyxrETGKu2GRASPs2H8zZAARifPA&usqp=CAU", 
                caption="Deteksi Penyakit Mata Bertenaga AI")
        
        st.markdown("""
        <div class="info-box">
        <strong>⚠️ Penafian Medis</strong><br>
        Sistem ini hanya untuk tujuan edukasi dan penelitian. 
        Selalu konsultasikan dengan profesional kesehatan yang qualified untuk diagnosis dan pengobatan medis.
        </div>
        """, unsafe_allow_html=True)

def detection_page():
    """Disease detection page"""
    st.markdown('<h2 class="subheader">🔍 Deteksi Penyakit Mata</h2>', unsafe_allow_html=True)
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Memuat model AI..."):
            st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Tidak dapat memuat model. Silakan periksa apakah file model ada.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Unggah Gambar Fundus")
        uploaded_file = st.file_uploader(
            "Pilih gambar fundus...", 
            type=['png', 'jpg', 'jpeg'],
            help="Unggah gambar fundus (retina) yang jelas untuk analisis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
            
            # Image information
            st.markdown("**Detail Gambar:**")
            st.write(f"- Ukuran: {image.size}")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- Format: {uploaded_file.type}")
            
            # Predict button
            if st.button("🔬 Analisis Gambar", type="primary", use_container_width=True):
                with st.spinner("Menganalisis gambar..."):
                    predicted_class, confidence, probabilities = predict_disease(
                        st.session_state.model, image, st.session_state.class_names
                    )
                    
                    if predicted_class is not None:
                        # Store in history
                        st.session_state.prediction_history.append({
                            'image_name': uploaded_file.name,
                            'prediction': predicted_class,
                            'confidence': confidence,
                            'timestamp': pd.Timestamp.now()
                        })
                        
                        # Display results in col2
                        with col2:
                            display_prediction_results(predicted_class, confidence, probabilities)
    
    with col2:
        st.markdown("### 📋 Petunjuk")
        st.markdown("""
        1. **Unggah Gambar**: Pilih gambar fundus yang jelas
        2. **Tunggu Analisis**: AI memproses gambar
        3. **Lihat Hasil**: Dapatkan prediksi dengan kepercayaan
        4. **Baca Informasi**: Pelajari tentang kondisi yang terdeteksi
        
        **Persyaratan Gambar:**
        - Gambar fundus/retina yang jelas
        - Pencahayaan dan fokus yang baik
        - Fotografi fundus standar
        - Format yang didukung: PNG, JPG, JPEG
        """)
        
        # Show prediction history
        if st.session_state.prediction_history:
            st.markdown("### 📈 Prediksi Terbaru")
            history_df = pd.DataFrame(st.session_state.prediction_history[-5:])  # Last 5 predictions
            st.dataframe(history_df[['image_name', 'prediction', 'confidence']], use_container_width=True)

def display_prediction_results(predicted_class, confidence, probabilities):
    """Display prediction results"""
    st.markdown("### 🎯 Hasil Analisis")
    
    # Main prediction
    st.markdown(f"""
    <div class="prediction-box">
    <h3>Kondisi Terdeteksi: <strong>{predicted_class}</strong></h3>
    <h4>Kepercayaan: <strong>{confidence:.2%}</strong></h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence indicator
    if confidence > 0.8:
        st.success("Prediksi kepercayaan tinggi")
    elif confidence > 0.6:
        st.warning("Prediksi kepercayaan sedang")
    else:
        st.error("Prediksi kepercayaan rendah - pertimbangkan konsultasi dengan spesialis")
    
    # Probability chart
    st.plotly_chart(create_probability_chart(probabilities), use_container_width=True)
    
    # Disease information
    disease_info = get_disease_info(predicted_class)
    if disease_info:
        st.markdown("### 📚 Informasi Kondisi")
        
        with st.expander(f"Pelajari lebih lanjut tentang {predicted_class}", expanded=True):
            st.markdown(f"**Deskripsi:** {disease_info['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Gejala Umum:**")
                for symptom in disease_info['symptoms']:
                    st.write(f"• {symptom}")
            
            with col2:
                st.markdown("**Pilihan Pengobatan:**")
                st.write(disease_info['treatment'])
                
                st.markdown("**Pencegahan:**")
                st.write(disease_info['prevention'])

def model_info_page():
    """Model information page"""
    st.markdown('<h2 class="subheader">🤖 Informasi Model</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Detail Arsitektur")
        st.markdown("""
        **Jenis Model yang Diimplementasikan:**
        1. **CNN Kustom**: Dibangun dari awal dengan arsitektur yang ditingkatkan dengan teknik modern
        2. **Transfer Learning VGG16**: Pra-dilatih pada ImageNet
        
        **Fitur Utama:**
        - Batch Normalization untuk pelatihan yang stabil
        - Lapisan Dropout untuk regularisasi
        - Early Stopping untuk mencegah overfitting
        - Augmentasi data untuk generalisasi yang lebih baik
        """)
    
    with col2:
        st.markdown("### 📊 Konfigurasi Pelatihan")
        st.markdown("""
        **Pembagian Dataset:**
        - Pelatihan: 70%
        - Validasi: 15% 
        - Pengujian: 15%
        
        **Hyperparameter:**
        - Ukuran Gambar: 224x224 piksel
        - Ukuran Batch: 64
        - Optimizer: Adam
        - Fungsi Loss: Sparse Categorical Crossentropy
        - Epochs: Hingga 50 (dengan early stopping)
        """)
    
    st.markdown("### 🎯 Perbandingan Kinerja Model")
    
    # Sample performance data (replace with actual results)
    performance_data = {
        'Model': ['CNN Kustom', 'Transfer VGG16'],
        'Akurasi Test': [0.7607, 0.8938],
        'Waktu Pelatihan (menit)': [45, 30],
        'Parameter (Juta)': [26.3, 15.1]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    # Performance chart
    fig = px.bar(performance_df, x='Model', y='Akurasi Test', 
                title="Perbandingan Kinerja Model",
                color='Akurasi Test',
                color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)

def dataset_stats_page():
    """Dataset statistics page"""
    st.markdown('<h2 class="subheader">📊 Statistik Dataset</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    dataset_info = {
        'Kelas': ['Katarak', 'Retinopati Diabetik', 'Glaukoma', 'Normal', 'Total'],
        'Gambar': [1038, 1098, 1007, 1074, 4217],
        'Persentase': [24.6, 26.0, 23.9, 25.5, 100.0]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Distribusi Kelas")
        df = pd.DataFrame(dataset_info)
        st.dataframe(df, use_container_width=True)
        
        # Pie chart
        fig_pie = px.pie(df[:-1], values='Gambar', names='Kelas', 
                        title="Distribusi Kelas Dataset")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Detail Dataset")
        st.markdown("""
        **Spesifikasi Gambar:**
        - Resolusi: 512x512 piksel
        - Format: Berbagai (PNG, JPG, JPEG)
        - Warna: RGB (3 channel)
        - Ukuran Total: 4.217 gambar
        
        **Kualitas Data:**
        - Gambar fundus beresolusi tinggi
        - Fotografi medis profesional
        - Distribusi kelas yang seimbang
        - Anotasi medis terverifikasi
        
        **Augmentasi yang Diterapkan:**
        - Rotasi acak (±15°)
        - Flipping horizontal
        - Variasi brightness
        - Normalisasi (skala 0-1)
        """)
        
        # Bar chart
        fig_bar = px.bar(df[:-1], x='Kelas', y='Gambar',
                        title="Gambar per Kelas",
                        color='Gambar',
                        color_continuous_scale='blues')
        st.plotly_chart(fig_bar, use_container_width=True)

def about_page():
    """About page"""
    st.markdown('<h2 class="subheader">ℹ️ Tentang Sistem Ini</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Gambaran Proyek
        Sistem Deteksi Penyakit Mata Fundus ini adalah alat bertenaga AI yang dirancang untuk membantu dalam 
        deteksi dini penyakit mata umum menggunakan fotografi fundus (retina). Sistem ini menggunakan teknik 
        pembelajaran mendalam canggih untuk menganalisis gambar retina dan memberikan klasifikasi yang akurat.
        
        ### 🔬 Stack Teknologi
        - **Framework Deep Learning**: TensorFlow/Keras
        - **Arsitektur CNN**: CNN Kustom, VGG16
        - **Framework Web**: Streamlit
        - **Pemrosesan Gambar**: OpenCV, PIL
        - **Visualisasi Data**: Plotly, Matplotlib
        - **Lingkungan Pengembangan**: Google Colab, VS Code
        
        ### 🎓 Tujuan Edukasi
        Sistem ini dikembangkan untuk tujuan edukasi dan penelitian untuk mendemonstrasikan:
        - Implementasi CNN untuk klasifikasi gambar medis
        - Teknik transfer learning dalam AI kesehatan
        - Metode perbandingan dan evaluasi model
        - Deployment model AI menggunakan Streamlit
        
        ### 📚 Informasi Dataset
        Model dilatih pada dataset komprehensif yang berisi 4.217 gambar fundus berkualitas tinggi 
        dalam empat kategori: Katarak, Retinopati Diabetik, Glaukoma, dan mata Normal.
        """)
    
    with col2:
        st.markdown("""
        ### 👨‍💻 Info Pengembangan
        **Dibuat oleh**: Fadilah Kurniawan Hadi  
        **Versi**: 1.0   
        **Terakhir Diperbarui**: 2025             
        **Lisensi**: Penggunaan Edukasi  
        
        ### 🔗 Fitur Utama
        - Beragam arsitektur CNN
        - Analisis gambar real-time
        - Penilaian kepercayaan
        - Informasi penyakit
        - Metrik kinerja
        - Interface ramah pengguna
        
        ### ⚠️ Pemberitahuan Penting
        Sistem ini ditujukan untuk tujuan edukasi 
        dan penelitian saja. Tidak boleh digunakan 
        sebagai pengganti diagnosis atau pengobatan 
        medis profesional.
        
        Selalu konsultasikan dengan profesional 
        kesehatan yang qualified untuk saran medis.
        """)
        
        st.markdown("""
        <div class="info-box">
        <strong>📧 Kontak</strong><br>
        Untuk pertanyaan atau feedback tentang sistem ini, 
        silakan konsultasikan dengan instruktur atau tim 
        pengembangan Anda.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()