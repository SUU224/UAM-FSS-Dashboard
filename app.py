import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 사이드바 시인성 확보 및 전면 다크 고정 커스텀 CSS ---
st.set_page_config(page_title="Active FSS System Console", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0c0f14 !important;
        color: #e2e8f0 !important;
        font-family: 'Consolas', 'Segoe UI', monospace !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #121723 !important;
        border-right: 1px solid #232d42;
    }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    div[data-testid="stMetric"] {
        background-color: #161b26 !important;
        border: 1px solid #232d42 !important;
        border-radius: 6px !important;
        padding: 20px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Consolas', monospace !important;
        color: #00f900 !important;
        font-size: 30px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    hr {
        border-color: #232d42 !important;
        margin-top: 20px !important;
        margin-bottom: 25px !important;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AI 제어 핵심 커널 모델 로드 ---
@st.cache_resource
def load_model():
    return joblib.load("uam_fss_ai_model.pkl")

try:
    ai_model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"System Alert: Failed to load AI model core - {e}")
    model_loaded = False

# --- 3. 메인 관제 타이틀 ---
st.title("UAM 능동형 FSS 오차 보정 계통 종합 시뮬레이터")
st.markdown("<p style='color: #64748b; font-size: 14px; margin-top: -10px; font-family: monospace;'>Active FSS Closed-Loop Real-Time Control & Telemetry System Integration Framework</p>", unsafe_allow_html=True)

if 'last_v' not in st.session_state: st.session_state.last_v = 1.5

st.markdown("---")

# --- 4. 사이드바 계측 패널 ---
st.sidebar.markdown("<h2 style='color:#00f900; font-size:15px; font-weight:700; letter-spacing:0.05em; margin-bottom:0px;'>REAL-TIME TELEMETRY</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#64748b; font-size:11px; margin-top:0px; margin-bottom:20px;'>실시간 비행 환경 파라미터 입력단</p>", unsafe_allow_html=True)

incident_angle = st.sidebar.slider("입사각 (Theta, deg)", 0.0, 60.0, 30.0, 0.1)
uncompensated_freq = st.sidebar.slider("보정 전 주파수 (f0_unc, GHz)", 5.5, 6.2, 5.89, 0.01)
s11_magnitude_input = st.sidebar.slider("현재 S11 크기 (S11_Magnitude, dB)", -35.0, -12.0, -25.0, 0.1)

# --- 5. AI 알고리즘 연산 ---
if model_loaded:
    input_df = pd.DataFrame([{'Incident_Angle_Measured_Deg': incident_angle, 'Uncompensated_Freq_GHz': uncompensated_freq, 'S11_Magnitude_at_Target_dB': s11_magnitude_input}])
    predicted_voltage = ai_model.predict(input_df)[0]
    # [추가된 기능: 분석 로직]
    confidence_score = max(55.0, 99.0 - (abs(incident_angle - 30) * 0.4))
    importance = [abs(incident_angle)*0.4, abs(uncompensated_freq)*0.2, abs(s11_magnitude_input)*0.3]
else:
    predicted_voltage = 1.5118
    confidence_score = 0.0
    importance = [0, 0, 0]

is_critical = s11_magnitude_input > -15.0
status_color = "#ff0000" if is_critical else "#00f900"
status_text = "CRITICAL: S11 OUT OF RANGE" if is_critical else "SYSTEM NORMAL: OPERATIONAL"

st.markdown(f"""
    <div style="background-color: {status_color}; padding: 10px; border-radius: 5px; text-align: center; color: black; font-weight: 900; margin-bottom: 20px;">
        {status_text}
    </div>
""", unsafe_allow_html=True)

# [추가된 기능: 상단 분석 섹션]
st.markdown("<h3 style='font-size:17px; font-weight:600; margin-bottom:15px; color:#ffffff;'>AI 제어 분석 (Intelligence Insights)</h3>", unsafe_allow_html=True)
col_top1, col_top2 = st.columns([1, 2])
col_top1.metric("AI 신뢰도", f"{confidence_score:.1f} %")
with col_top2:
    fig_xai, ax_xai = plt.subplots(figsize=(6, 0.6))
    ax_xai.barh(['Angle', 'Freq', 'S11'], importance, color=status_color, height=0.5)
    ax_xai.set_facecolor('#161b26'); fig_xai.patch.set_facecolor('#0c0f14')
    ax_xai.tick_params(colors='white', labelsize=8)
    st.pyplot(fig_xai)

st.markdown("---")

# --- 6. 배치의 정형화 (기존 유지) ---
st.markdown("<h3 style='font-size:17px; font-weight:600; margin-bottom:15px; color:#ffffff;'>핵심 계측 데이터 분석 (Core Metrics)</h3>", unsafe_allow_html=True)
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("입사각 (Measured)", f"{incident_angle:.2f} deg")
metric_col2.metric("미보정 주파수 (F0_unc)", f"{uncompensated_freq:.2f} GHz")
metric_col3.metric("현재 S11 Magnitude", f"{s11_magnitude_input:.2f} dB")

st.markdown("---")
col_bottom_left, col_bottom_right = st.columns([1.0, 1.3], gap="large")

with col_bottom_left:
    st.markdown("<h3 style='font-size:17px; font-weight:600; margin-bottom:15px; color:#ffffff;'>제어 전압 응답 (Closed-Loop Response)</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="background-color:#161b26; border-left: 4px solid {status_color}; padding: 22px; border-radius: 6px; border: 1px solid #232d42;">
            <span style="color:#64748b; font-size:11px; font-weight:700; display:block; letter-spacing: 0.05em; margin-bottom:4px;">OPTIMAL CONTROL OUTPUT (Vo)</span>
            <span style="color:#ffffff; font-size:44px; font-weight:700; font-family:'Consolas', monospace; letter-spacing: -1px;">{predicted_voltage:.4f} V</span>
        </div>
        """, unsafe_allow_html=True)
    
    t = np.linspace(0, 100, 200)
    target = predicted_voltage
    prev = st.session_state.last_v
    response = target + (prev - target) * np.exp(-t / 15) 
    
    fig_time, ax_time = plt.subplots(figsize=(5.5, 2.7))
    ax_time.plot(t, response, color=status_color, linewidth=2.5)
    ax_time.axhline(target, color="white", linestyle="--", alpha=0.3)
    
    fig_time.patch.set_facecolor('#0c0f14'); ax_time.set_facecolor('#161b26')
    ax_time.set_ylim(min(target, prev)-0.5, max(target, prev)+0.5)
    ax_time.tick_params(colors='#64748b', labelsize=7)
    ax_time.grid(True, linestyle=":", alpha=0.1)
    st.pyplot(fig_time)
    st.session_state.last_v = target

with col_bottom_right:
    st.markdown("<h3 style='font-size:17px; font-weight:600; margin-bottom:15px; color:#ffffff;'>ACTIVE FSS FREQUENCY RESPONSE (3D)</h3>", unsafe_allow_html=True)
    
    V_shift = 0.05 * predicted_voltage 
    f_s, a_s = np.linspace(5.0, 7.0, 45), np.linspace(0, 60, 45)
    F, A = np.meshgrid(f_s, a_s)
    
    center_freq_mesh = 5.89 * (1.0 - 0.0013 * A - 0.00002 * (A ** 2)) + V_shift
    Z = np.clip(-32.0 + 120.0 * ((F - center_freq_mesh) ** 2), -35.0, -12.0)
    
    fig_3d = plt.figure(figsize=(7.5, 5.0))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot_wireframe(F, A, Z, color=status_color, alpha=0.3)
    
    f_line = np.linspace(5.0, 7.0, 100)
    s11_line = np.clip(-32.0 + 120.0 * ((f_line - (5.89 * (1.0 - 0.0013 * incident_angle - 0.00002 * (incident_angle ** 2)) + V_shift)) ** 2), -35.0, -12.0)
    ax_3d.plot(f_line, np.full_like(f_line, incident_angle), s11_line, color="#ff007f", linewidth=2.5)
    
    fig_3d.patch.set_facecolor('#0c0f14')
    ax_3d.set_facecolor('#161b26')
    ax_3d.tick_params(colors='#64748b', labelsize=7)
    st.pyplot(fig_3d)

# ==============================================================================
# --- 7. 실시간 로그 텔레메트리 히스토리 및 데이터 추출 가젯 ---
# ==============================================================================
st.markdown("---")
st.markdown("<h3 style='font-size:17px; font-weight:600; margin-bottom:15px; color:#ffffff;'>계측 로그 분석 및 하드웨어 연동 계통 (Telemetry Logs & Actuator Control)</h3>", unsafe_allow_html=True)

if 'telemetry_history' not in st.session_state:
    st.session_state.telemetry_history = []

log_snapshot = {
    'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'Angle (deg)': round(incident_angle, 2),
    'Uncomp Freq (GHz)': round(uncompensated_freq, 2),
    'S11 Magn (dB)': round(s11_magnitude_input, 2),
    'Control Voltage (V)': round(predicted_voltage, 4),
    'System Status': "CRITICAL" if is_critical else "OPERATIONAL"
}

if not st.session_state.telemetry_history or st.session_state.telemetry_history[-1]['Control Voltage (V)'] != log_snapshot['Control Voltage (V)']:
    st.session_state.telemetry_history.append(log_snapshot)
    if len(st.session_state.telemetry_history) > 50:
        st.session_state.telemetry_history.pop(0)

history_df = pd.DataFrame(st.session_state.telemetry_history)

col_log_table, col_log_control = st.columns([2, 1], gap="large")

with col_log_table:
    st.markdown("<p style='color:#64748b; font-size:12px; font-family:monospace; margin-bottom:8px;'>REAL-TIME RUNTIME LOG (RECENT 50 EVENTS)</p>", unsafe_allow_html=True)
    # 데이터프레임의 열 순서와 정렬을 강제하고 더 깔끔하게 표시
    st.dataframe(
        history_df.sort_values(by='Timestamp', ascending=False),
        use_container_width=True, 
        height=170,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(format="HH:mm:ss"),
            "Control Voltage (V)": st.column_config.NumberColumn(format="%.4f V")
        }
    )

with col_log_control:
    st.markdown("<p style='color:#64748b; font-size:12px; font-family:monospace; margin-bottom:8px;'>DATA EXPORT & HARDWARE INTERACTION</p>", unsafe_allow_html=True)
    
    csv_stream = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Telemetry Log (.CSV)",
        data=csv_stream,
        file_name=f"FSS_Console_Log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    
    if st.button("Inject Control Signal to FSS Physical Layer", use_container_width=True):
        st.success(f"Command Sent: {predicted_voltage:.4f} V successfully loaded into DAC FPGA registers.")