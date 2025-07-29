# kidney_app.py
import streamlit as st
import pandas as pd
import numpy as np

import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import shutil
import tempfile
import datetime
import warnings
from fpdf import FPDF
from PIL import Image
import zipfile
import io

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Chiagoziem's Kidney Disease Risk Assessment App",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for styling and auto dark mode via prefers-color-scheme
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #000000;
    }
    @media (prefers-color-scheme: dark) {
        body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stApp {
            background-color: #1e1e1e;
        }
    }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton > button, .stDownloadButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #0051a3;
        color: white;
    }
    .st-expander > summary {
        transition: all 0.3s ease;
    }
    .st-expander > summary:hover {
        color: #0066cc;
    }
    header[data-testid="stHeader"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR THEME ====================
with st.sidebar:
    st.title("🩺 Kidney Risk App")
    st.markdown("""
    Welcome to Chiagoziem's **Kidney Disease Risk Assessment App**. Use the tabs to assess individual risk or upload patient data.
    """)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_models():
    models = {
        'ckd_model': joblib.load('ckd_model.pkl'),
        'dialysis_model': joblib.load('dialysis_model.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }
    return models

# ==================== UTILITY FUNCTIONS ====================
def create_temp_directory():
    temp_dir = tempfile.mkdtemp()
    st.session_state['temp_dirs'] = st.session_state.get('temp_dirs', []) + [temp_dir]
    return temp_dir

def cleanup_temp_directories():
    if 'temp_dirs' in st.session_state:
        for temp_dir in st.session_state['temp_dirs']:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                st.warning(f"⚠️ Could not remove temporary directory {temp_dir}: {e}")
        st.session_state['temp_dirs'] = []

def zip_reports(directory_path, zip_name="All_Patient_Reports.zip"):
    zip_path = os.path.join(directory_path, zip_name)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir(directory_path):
            if file.endswith(".pdf"):
                zipf.write(os.path.join(directory_path, file), arcname=file)
    return zip_path

def generate_risk_visuals(ckd_prob, dialysis_prob, path):
    fig = go.Figure(data=[
        go.Bar(
            x=['CKD Risk', 'Dialysis Risk'],
            y=[ckd_prob, dialysis_prob],
            text=[f"{ckd_prob:.1%}", f"{dialysis_prob:.1%}"],
            textposition='auto',
            marker_color=['#ff6361', '#58508d']
        )
    ])
    fig.update_layout(
        title="Risk Visualization",
        yaxis=dict(range=[0, 1]),
        height=400,
        margin=dict(l=0, r=0, t=40, b=40),
        plot_bgcolor='white'
    )
    fig_path = os.path.join(path, "risk_plot.png")
    fig.write_image(fig_path)
    return fig_path

# ==================== PDF REPORT CLASS ====================
class MedicalPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_font('Helvetica', '', 12)

    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Kidney Disease Risk Assessment Report', 0, 1, 'C')
        self.set_font('Helvetica', '', 12)
        self.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_colored_image(self, image_path, x=None, w=190):
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    rgb_path = os.path.splitext(image_path)[0] + '_rgb.png'
                    img.save(rgb_path, format='PNG')
                    image_path = rgb_path
                self.image(image_path, x=x, w=w)
                self.ln(5)
            except Exception as e:
                st.warning(f"⚠️ Could not add image to PDF: {e}")

    def add_patient_info(self, name, ckd_prob, dialysis_prob):
        self.set_font('Helvetica', '', 12)
        self.cell(0, 10, f"Patient Name: {name}", 0, 1)
        self.cell(0, 10, f"CKD Risk: {ckd_prob:.1%}", 0, 1)
        self.cell(0, 10, f"Dialysis Risk: {dialysis_prob:.1%}", 0, 1)
        interpretation = "High risk: Recommend immediate clinical evaluation." if ckd_prob > 0.7 or dialysis_prob > 0.7 else "Moderate/low risk: Recommend routine monitoring."
        self.multi_cell(0, 10, f"Interpretation: {interpretation}")
        self.ln(5)

    def add_features_table(self, data_dict):
        self.set_font('Helvetica', '', 11)
        self.ln(3)
        self.set_fill_color(230, 230, 230)
        for key, value in data_dict.items():
            self.cell(60, 10, str(key), 1, 0, 'L', fill=True)
            self.cell(0, 10, str(value), 1, 1, 'L')
        self.ln(5)

# ==================== RISK ASSESSMENT ====================
def risk_assessment(models):
    st.header("🧪 Individual Risk Assessment")

    with st.form(key="risk_form"):
        name = st.text_input("Patient Name", help="Enter full name of the patient")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, help="Patient's age in years")
            creatinine = st.number_input("Creatinine Level", 0.0, 20.0, step=0.1)
            bun = st.number_input("BUN", 0.0, 100.0, step=0.1)
        with col2:
            diabetes = st.selectbox("Diabetes", ["Yes", "No"])
            hypertension = st.selectbox("Hypertension", ["Yes", "No"])
            gfr = st.number_input("GFR", 0.0, 120.0, step=0.1)
            urine = st.number_input("Urine Output (ml/day)", min_value=0.0, step=0.1)

        submit = st.form_submit_button("🔍 Assess Risk")

    if submit:
        diabetes_bin = 1 if diabetes == 'Yes' else 0
        hypertension_bin = 1 if hypertension == 'Yes' else 0

        input_data = np.array([[age, creatinine, bun, diabetes_bin, hypertension_bin, gfr, urine]])
        input_scaled = models['scaler'].transform(input_data)

        ckd_prob = models['ckd_model'].predict_proba(input_scaled)[0][1]
        dialysis_prob = models['dialysis_model'].predict_proba(input_scaled)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("CKD Risk")
            st.metric("Probability", f"{ckd_prob:.1%}")
            fig1 = px.bar(x=['Low Risk', 'High Risk'],
                         y=[1-ckd_prob, ckd_prob],
                         color=['Low Risk', 'High Risk'],
                         color_discrete_sequence=['green', 'red'],
                         labels={'x': 'Risk Level', 'y': 'Probability'},
                         height=300)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.subheader("Dialysis Need")
            st.metric("Probability", f"{dialysis_prob:.1%}")
            fig2 = px.bar(x=['Not Needed', 'Needed'],
                         y=[1-dialysis_prob, dialysis_prob],
                         color=['Not Needed', 'Needed'],
                         color_discrete_sequence=['green', 'orange'],
                         labels={'x': 'Status', 'y': 'Probability'},
                         height=300)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🩻 Interpretation")
        if ckd_prob > 0.7:
            st.error("High risk of CKD — Clinical evaluation recommended.")
        elif ckd_prob > 0.3:
            st.warning("Moderate risk — Monitor closely.")
        else:
            st.success("Low CKD risk.")

        if dialysis_prob > 0.5:
            st.error("High chance of dialysis needed — Refer to nephrologist.")
        elif dialysis_prob > 0.2:
            st.warning("Moderate dialysis risk — Monitor regularly.")
        else:
            st.success("Low probability of dialysis need.")

        tmp_dir = create_temp_directory()
        chart_path = generate_risk_visuals(ckd_prob, dialysis_prob, tmp_dir)
        pdf = MedicalPDF()
        pdf.add_page()
        pdf.add_patient_info(name, ckd_prob, dialysis_prob)
        features = {
            'Age': age,
            'Creatinine_Level': creatinine,
            'BUN': bun,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'GFR': gfr,
            'Urine_Output': urine
        }
        pdf.add_features_table(features)
        pdf.add_colored_image(chart_path)
        file_path = os.path.join(tmp_dir, f"{name}_report.pdf")
        pdf.output(file_path)

        with open(file_path, "rb") as f:
            st.download_button("📄 Download Patient Report (PDF)", f, file_name=f"{name}_report.pdf")

# ==================== DATA EXPLORER ====================
def data_explorer(models):
    st.header("📂 Batch Data Explorer")
    uploaded_file = st.file_uploader("Upload Patient Excel or CSV File", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = df.drop(columns=[col for col in ['CKD_Status', 'Dialysis_Needed'] if col in df.columns])

        diabetes_map = {'Yes': 1, 'No': 0, 1: 1, 0: 0}
        df['Diabetes'] = df['Diabetes'].map(diabetes_map)
        df['Hypertension'] = df['Hypertension'].map(diabetes_map)

        input_data = df[['Age', 'Creatinine_Level', 'BUN', 'Diabetes', 'Hypertension', 'GFR', 'Urine_Output']]
        scaled = models['scaler'].transform(input_data)
        df['CKD_Risk'] = models['ckd_model'].predict_proba(scaled)[:, 1]
        df['Dialysis_Risk'] = models['dialysis_model'].predict_proba(scaled)[:, 1]

        st.dataframe(df)

        st.subheader("📊 Exploratory Data Analysis")
        with st.expander("Feature Distributions"):
            for column in ['Age', 'Creatinine_Level', 'BUN', 'GFR', 'Urine_Output']:
                fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}", color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("CKD and Dialysis Risk Correlations"):
            fig = px.scatter(df, x='CKD_Risk', y='Dialysis_Risk', color='Age', title="CKD Risk vs Dialysis Risk", color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)

        tmp_dir = create_temp_directory()
        for i, row in df.iterrows():
            pdf = MedicalPDF()
            pdf.add_page()
            name = row.get("Name", f"Patient_{i+1}")
            pdf.add_patient_info(name, row['CKD_Risk'], row['Dialysis_Risk'])
            features = row[['Age', 'Creatinine_Level', 'BUN', 'Diabetes', 'Hypertension', 'GFR', 'Urine_Output']].to_dict()
            pdf.add_features_table(features)
            chart_path = generate_risk_visuals(row['CKD_Risk'], row['Dialysis_Risk'], tmp_dir)
            pdf.add_colored_image(chart_path)
            pdf_path = os.path.join(tmp_dir, f"{name}_report.pdf")
            pdf.output(pdf_path)

        zip_path = zip_reports(tmp_dir)
        with open(zip_path, "rb") as f:
            st.download_button("📦 Download All Patient Reports (ZIP)", f, file_name="All_Patient_Reports.zip")

# ==================== MAIN ====================
def main():
    models = load_models()
    tabs = st.tabs(["🧪 Risk Assessment", "📂 Data Explorer", "ℹ️ Disclaimer"])
    with tabs[0]:
        risk_assessment(models)
    with tabs[1]:
        data_explorer(models)
    with tabs[2]:
        st.subheader("Disclaimer")
        st.info("This application is intended for educational and informational purposes only. It does not constitute medical advice.")

if __name__ == "__main__":
    main()
    cleanup_temp_directories()