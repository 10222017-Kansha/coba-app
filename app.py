import streamlit as st
import json
import os
import base64
from pathlib import Path

# --- IMPORT FUNGSI DARI BACKEND ---
# Pastikan file backend.py ada di folder yang sama
import backend 

# ==========================================
# 0. FUNGSI UTILITAS GAMBAR
# ==========================================
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

bg_image_file = "Dicoding.png"
spinner_image_file = "Dicoding (1).png"

# ==========================================
# 1. KONFIGURASI & CSS
# ==========================================
st.set_page_config(page_title="AI Interview Assessor", page_icon="🤖", layout="wide")

bg_image_b64 = img_to_bytes(bg_image_file) if os.path.exists(bg_image_file) else ""
spinner_image_b64 = img_to_bytes(spinner_image_file) if os.path.exists(spinner_image_file) else ""

st.markdown(f"""
<style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image_b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .stButton > button {{
        background-color: #2e4073; color: white; border-radius: 8px;
        padding: 12px 24px; font-size: 18px; font-weight: bold; border: none; width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: 0.3s;
    }}
    .stButton > button:hover {{ background-color: #1e2b4d; transform: translateY(-2px); }}
    [data-testid="stFileUploader"] {{
        background-color: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    h1, h2, h3, p {{ font-family: 'Segoe UI', sans-serif; color: #2e4073; }}
    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    .loading-spinner {{
        display: block; margin-left: auto; margin-right: auto; width: 80px;
        animation: spin 2s linear infinite;
    }}
    .loading-text {{ text-align: center; color: #2e4073; font-size: 18px; margin-top: 15px; font-weight: 500; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MAIN UI
# ==========================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h3 style='text-align: center;'>KELOMPOK A25-C351 | Dicoding</h3>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>AI Powered Interview System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    div.stButton > button {
        color: #000000;              
        background-color: #FFFFFF; 
        border: 2px solid black;
        padding: 10px 24px;
        border-radius: 5px;
    }
    
    
    div.stButton > button:hover {
        background-color: #FFFFFF; 
        color: #000000;
    }

    
    div.stButton {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Silahkan unggah file berformat .json di sini", type=['json'])

    if uploaded_file is None:
        st.info("Silakan unggah file JSON terlebih dahulu sebelum memulai analisis.")
    else:
        if uploaded_file.name != "payload.json":
            st.error("Nama file harus 'payload.json'. Silakan unggah file yang benar.")
        else:
            try:
                input_data = json.load(uploaded_file)
                # Panggil fungsi backend
                video_links = backend.parse_input_json(input_data)
                
                if not video_links:
                    st.error("File JSON tidak mengandung link video Google Drive.")
                else:
                    if 'done' not in st.session_state: st.session_state['done'] = False

                    if not st.session_state['done']:
                        start_btn = st.button("Mulai analisis")
                        
                        if start_btn:
                            loading_placeholder = st.empty()
                            loading_placeholder.markdown(f"""
                                <div style="margin-top: 20px; margin-bottom: 20px;">
                                    <img src="data:image/png;base64,{spinner_image_b64}" class="loading-spinner">
                                    <div class="loading-text">File sedang dianalisis...</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            try:
                                # --- PANGGIL FUNGSI BACKEND ---
                                audios, videos = backend.process_videos_pipeline(video_links)
                                transcripts = backend.transcribe_audios(audios)
                                scores, reasons = backend.grade_answers(transcripts, 'assessment_metric.json')
                                final_report = backend.generate_final_report_v2(input_data, transcripts, scores, reasons)
                                
                                st.session_state['result'] = final_report
                                st.session_state['done'] = True
                                
                            except Exception as e:
                                st.error(f"Terjadi kesalahan sistem: {str(e)}")
                            
                            loading_placeholder.empty()

            except json.JSONDecodeError:
                st.error("File yang diunggah tidak berformat JSON.")

    if st.session_state.get('done'):
        st.success("Analisis Selesai!")
        json_str = json.dumps(st.session_state['result'], indent=2)
        st.download_button(
            label="Download hasil analisis",
            data=json_str,
            file_name="final_assessment_report.json",
            mime="application/json"
        )