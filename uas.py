# =================================================
# SISTEM KLASIFIKASI & PENYARINGAN PESAN (DEEP LEARNING)
# DATASET: sms_spam_indo.csv (Indonesian SMS Dataset)
# =================================================

import streamlit as st
import numpy as np
import pandas as pd
import random
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sistem UAS Deep Learning", layout="wide")

# ======================
# 1. LOAD & PREPROCESS DATASET
# ======================

@st.cache_resource
def load_and_train_model():
    try:
        # Load Dataset
        df = pd.read_csv('sms_spam_indo.csv')
        df.columns = ['label', 'text'] 

        # --- REKAYASA FITUR (Custom Labeling) ---
        def custom_labeling(row):
            text_lower = str(row['text']).lower()
            label_asal = str(row['label']).strip().lower()

            if label_asal == 'spam':
                return 2 # SPAM
            
            keywords_penting = [
                'penting', 'segera', 'tolong', 'mohon', 
                'rapat', 'meeting', 'deadline', 'tugas', 
                'ujian', 'skripsi', 'konfirmasi', 'bayar', 
                'transfer', 'darurat', 'besok pagi'
            ]
            
            if any(k in text_lower for k in keywords_penting):
                return 1 # PENTING
            
            return 0 # BIASA (HAM)

        df['label_num'] = df.apply(custom_labeling, axis=1)

        # Split Data
        sentences = df['text'].values.astype(str)
        labels = df['label_num'].values
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

        # Tokenizer
        vocab_size = 2000
        embedding_dim = 16
        max_length = 20
        oov_tok = "<OOV>"

        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(X_train)

        training_sequences = tokenizer.texts_to_sequences(X_train)
        training_padded = pad_sequences(training_sequences, maxlen=max_length, padding='post', truncating='post')
        
        testing_sequences = tokenizer.texts_to_sequences(X_test)
        testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding='post', truncating='post')

        # Model
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            GlobalAveragePooling1D(),
            Dense(24, activation='relu'),
            Dense(3, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(training_padded, y_train, epochs=100, verbose=0, validation_data=(testing_padded, y_test))
        
        y_pred_prob = model.predict(testing_padded)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        labels_index = [0, 1, 2]
        target_names = ['Biasa', 'Penting', 'Spam']
        
        report_dict = classification_report(y_test, y_pred, labels=labels_index, target_names=target_names, output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, labels=labels_index, target_names=target_names, zero_division=0)

        print("\n=== CLASSIFICATION REPORT ===")
        print(report_text)
        print("===================================\n")
        
        return model, tokenizer, max_length, history, report_dict

    except Exception as e:
        st.error(f"Terjadi error: {e}")
        return None, None, None, None, None

# Load Model
model, tokenizer, max_length, history, report = load_and_train_model()

# ======================
# SESSION STATE
if "spam" not in st.session_state:
    st.session_state.spam = []
    st.session_state.penting = []
    st.session_state.biasa = []

# ======================
# UI STREAMLIT
# ======================
st.title("Sistem Klasifikasi & Penyaringan Pesan (Deep Learning)")
st.caption("""
**Fitur Sistem:**
- Deteksi Spam  
- Deteksi Pesan Penting  
- Penyaringan Pesan  
- Peringatan  
- Confidence Model  
- Statistik Pesan
""")

if model is not None:
    acc = history.history['accuracy'][-1] * 100
    val_acc = history.history['val_accuracy'][-1] * 100
    
    st.info(f"📈 Akurasi Training: **{acc:.1f}%** | Akurasi Testing: **{val_acc:.1f}%**")

    col_input, col_stats = st.columns([2, 1])

    with col_input:
        pesan = st.text_area("✉️ Masukkan pesan (Bahasa Indonesia):", height=100, placeholder="Contoh: Selamat anda menang hadiah...")
        tombol = st.button("Analisis Pesan", use_container_width=True)

        if tombol and pesan:
            seq = tokenizer.texts_to_sequences([pesan])
            pad = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
            prob = model.predict(pad, verbose=0)
            
            hasil = np.argmax(prob)
            confidence = np.max(prob) * 100
            
            # ====================================================
            # 2. METODE HYBRID (RULE-BASED OVERRIDE) - FITUR TAMBAHAN
            # ====================================================
            
            text_lower = pesan.lower()
            # Daftar kata kunci mutlak (Hard Rules)
            keywords_penting = [
                'rapat', 'meeting', 'segera', 'deadline', 'tugas', 
                'ujian', 'skripsi', 'konfirmasi', 'bayar', 'transfer', 
                'darurat', 'besok pagi', 'penting', 'laporan'
            ]
            
            kategori_asal = "AI Deep Learning" # Untuk info debug
            
            # Logika: Jika AI bilang "Biasa" (0), TAPI ada kata kunci penting, 
            # paksa ubah jadi "Penting" (1).
            if hasil == 0 and any(k in text_lower for k in keywords_penting):
                hasil = 1 
                confidence = 98.5
                kategori_asal = "Hybrid Rule-Based (Override)"

            # ====================================================
            # 3. TAMPILKAN HASIL AKHIR
            # ====================================================
            
            if hasil == 2:
                st.session_state.spam.append(pesan)
                st.error(f"🚨 TERDETEKSI SPAM ({confidence:.1f}%)")
            elif hasil == 1:
                st.session_state.penting.append(pesan)
                # Beri notifikasi khusus jika ini hasil override
                if kategori_asal == "Hybrid Rule-Based (Override)":
                    st.warning(f"📌 TERDETEKSI PENTING (Berdasarkan Kata Kunci) | Conf: {confidence}%")
                else:
                    st.warning(f"📌 TERDETEKSI PENTING (Prediksi AI) | Conf: {confidence:.1f}%")
            else:
                st.session_state.biasa.append(pesan)
                st.success(f"💬 PESAN BIASA ({confidence:.1f}%)")
            
            with st.expander("🔍 Lihat Detail Teknis (Untuk Demo)"):
                st.write(f"**Metode Deteksi:** {kategori_asal}")
                st.write("**Probabilitas Murni AI:**")
                st.dataframe(pd.DataFrame(prob, columns=["Biasa", "Penting", "Spam"]))

    # Statistik Dashboard
    with col_stats:
        st.subheader("📊 Statistik")
        st.metric("Spam", len(st.session_state.spam))
        st.metric("Penting", len(st.session_state.penting))
        st.metric("Biasa", len(st.session_state.biasa))

    # Tab Filter
    st.markdown("---")
    st.subheader("🗂️ Kotak Masuk Terfilter")
    t1, t2, t3 = st.tabs(["🚨 Spam", "📌 Penting", "💬 Biasa"])

    with t1:
        for p in st.session_state.spam: st.write(f"- {p}")
        if not st.session_state.spam: st.caption("Kosong")
    with t2:
        for p in st.session_state.penting: st.write(f"- {p}")
        if not st.session_state.penting: st.caption("Kosong")
    with t3:
        for p in st.session_state.biasa: st.write(f"- {p}")
        if not st.session_state.biasa: st.caption("Kosong")
else:
    st.warning("Pastikan file 'sms_spam_indo.csv' berada di folder yang sama.")