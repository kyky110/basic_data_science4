import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Load Preprocessing Components and Model ---
# Make sure these files are in the same directory as app.py or provide full paths
try:
    with open('scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        loaded_label_encoders = pickle.load(file)
    with open('one_hot_columns.pkl', 'rb') as file:
        # This file contains the list of *original* categorical columns that were one-hot encoded
        # not the resulting dummy column names. We need the list of resulting dummy column names.
        # Re-read the file to ensure correct content if the previous interpretation was off.
        original_one_hot_cols_names = pickle.load(file)

    with open('model_huber.pkl', 'rb') as file:
        loaded_model_huber = pickle.load(file)
    st.success("Model dan komponen preprocessing berhasil dimuat!")
except FileNotFoundError as e:
    st.error(f"Error: File tidak ditemukan - {e}. Pastikan file .pkl berada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat objek: {e}")
    st.stop()

# --- Define expected columns based on training for consistency ---
# This should match the X_train columns order and names exactly
expected_feature_order = [
    'Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
    'Jenis_Kelamin_L', 'Jenis_Kelamin_P', 'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
]

# --- Streamlit App Layout ---
st.title("Prediksi Gaji Awal Lulusan Pelatihan Vokasi")
st.markdown("Masukkan detail peserta untuk memprediksi gaji awal (dalam juta Rupiah).")

# --- User Inputs ---

st.subheader("Informasi Pribadi")
usia = st.number_input("Usia (tahun)", min_value=17.0, max_value=60.0, value=25.0, step=1.0)
jenis_kelamin_options = ['L', 'P'] # From df_bersih['Jenis_Kelamin'].unique()
jenis_kelamin = st.selectbox("Jenis Kelamin", options=jenis_kelamin_options)

st.subheader("Informasi Pendidikan & Pelatihan")
pendidikan_options = ['SMA', 'SMK', 'D3', 'S1'] # From df_bersih['Pendidikan'].unique()
pendidikan = st.selectbox("Pendidikan Terakhir", options=pendidikan_options)
jurusan_options = ['administrasi', 'teknik las', 'desain grafis', 'teknik listrik', 'otomotif'] # From df_bersih['Jurusan'].unique()
jurusan = st.selectbox("Jurusan Pelatihan Vokasi", options=jurusan_options)
durasi_jam = st.number_input("Durasi Pelatihan (jam)", min_value=30.0, max_value=100.0, value=60.0, step=1.0)
nilai_ujian = st.number_input("Nilai Ujian (skala 0-100)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)

st.subheader("Status Pekerjaan")
status_bekerja_options = ['Belum Bekerja', 'Sudah Bekerja'] # From df_bersih['Status_Bekerja'].unique()
status_bekerja = st.selectbox("Status Bekerja", options=status_bekerja_options)


# --- Prediction Button ---
if st.button("Prediksi Gaji Awal"):
    # --- 2. Create DataFrame from User Inputs ---
    new_data_raw = {
        'Usia': [usia],
        'Durasi_Jam': [durasi_jam],
        'Nilai_Ujian': [nilai_ujian],
        'Pendidikan': [pendidikan],
        'Jurusan': [jurusan],
        'Jenis_Kelamin': [jenis_kelamin],
        'Status_Bekerja': [status_bekerja]
    }
    new_df = pd.DataFrame(new_data_raw)

    # --- 3. Preprocessing New Data (Consistent with Training) ---

    # Label Encoding
    for col, le in loaded_label_encoders.items():
        if col in new_df.columns:
            new_df[col] = le.transform(new_df[col])

    # One-Hot Encoding
    # `original_one_hot_cols_names` contains ['Jenis_Kelamin', 'Status_Bekerja']
    new_df_onehot = pd.get_dummies(new_df[original_one_hot_cols_names])
    new_df_onehot = new_df_onehot.astype(int)

    # Reindex one-hot encoded columns to match the training data's structure
    # This is critical to ensure the correct number of columns and order.
    # The expected dummy column names are derived from `expected_feature_order`
    expected_one_hot_dummy_cols = [col for col in expected_feature_order if col.startswith(('Jenis_Kelamin_', 'Status_Bekerja_'))]
    new_df_onehot = new_df_onehot.reindex(columns=expected_one_hot_dummy_cols, fill_value=0)

    # Combine all features for scaling
    # Extract numerical and label-encoded columns first
    numerical_and_label_encoded_cols = [
        'Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan'
    ]
    processed_features = new_df[numerical_and_label_encoded_cols]
    processed_features = pd.concat([processed_features, new_df_onehot], axis=1)

    # Ensure the order of columns matches the training data (X_train)
    # If any column is missing (e.g., a specific one-hot category not present in user input),
    # it will be added with a value of 0 due to the reindex operation previously.
    # However, ensure that the final `processed_features` has all `expected_feature_order` columns in correct order
    final_features_df = processed_features.reindex(columns=expected_feature_order, fill_value=0)

    # Scaling numerical features
    # Only scale the numerical features. The scaler was fit on X_train, which was already fully prepared.
    # So we need to ensure 'final_features_df' is in the exact same format as X_train when applying transform.
    # The `loaded_scaler` expects all features that were part of `X_train`.
    scaled_features = loaded_scaler.transform(final_features_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=expected_feature_order)

    # --- 4. Make Prediction ---
    predicted_gaji = loaded_model_huber.predict(scaled_features_df)

    # --- 5. Display Result ---
    st.success(f"Prediksi Gaji Awal: **Rp {predicted_gaji[0]:.2f} Juta**")
    st.balloons()
