import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ================= Page Setup =================
st.set_page_config(page_title="Projectile Motion", layout="wide")
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>🎯 Interactive Projectile Simulation</h1>
<p style='text-align: center;'>Enter values, then press <b>🚀 Start Simulation</b>.</p>
<hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

# ================= Session State Setup =================
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "teacher_mode" not in st.session_state:
    st.session_state.teacher_mode = False
if "show_answer_result" not in st.session_state:
    st.session_state.show_answer_result = False
if "test_completed" not in st.session_state:
    st.session_state.test_completed = False
if "correct_answered" not in st.session_state:
    st.session_state.correct_answered = False
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False 

# ================= User Inputs =================
st.markdown("<h3 style='color:#154360;'>⚙️ Simulation Settings</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    v0 = st.number_input("🔹 Initial Velocity (m/s)", min_value=0.0, value=50.0, step=0.1, format="%.2f")
with col2:
    angle = st.number_input("🔹 Launch Angle (°)", min_value=0.0, max_value=90.0, value=45.0, step=0.1, format="%.2f")

col3, col4 = st.columns(2)
with col3:
    h0 = st.number_input("🔹 Initial Height (m)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
with col4:
    t_user = st.number_input("🔹 Instantaneous Time (s)", min_value=0.0, value=0.0, step=0.1, format="%.2f")

st.markdown("<p style='color:gray; font-size:16px;'>If Instantaneous Time = 0, no specific time analysis will be performed.</p>", unsafe_allow_html=True)

# ================= Additional Options =================
compare_angles = st.checkbox("📐 Compare Complementary Angles (θ & 90°−θ)")
air_resistance = st.checkbox("🌬️ Air Resistance")
compare_with_air = st.checkbox("🔄 Compare Trajectories (No Air vs Air Resistance)")

# ================= Physics and Calculations =================
g = 9.81
theta = np.radians(angle)

def calculate_flight_time(v0, theta_rad, h0, g):
    return (v0 * np.sin(theta_rad) + np.sqrt((v0*np.sin(theta_rad))**2 + 2*g*h0)) / g

t_flight = calculate_flight_time(v0, theta, h0, g)

if t_user > t_flight:
    st.warning("⏳ The entered time exceeds the flight time! The maximum possible time for the primary trajectory will be used.")
    t_user = t_flight

# ✅ تعديل: تقليل عدد النقاط لتسريع التحديث السلس
num_points = 200  

# ================= Colors Setup =================
trail_color = 'black'
bg_color = 'white'
if air_resistance:
    bg_color = '#00CED1'

# ================= Air Resistance Model =================
def compute_projectile_with_air(v0, theta, h0, g=9.81, c=0.005, dt=0.01):  
    t_points = [0]
    x_points = [0]
    y_points = [h0]
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    t = 0
    while y_points[-1] >= 0:
        v = np.sqrt(vx**2 + vy**2)
        ax = -c * v * vx
        ay = -g - c * v * vy
        vx += ax * dt
        vy += ay * dt
        x_points.append(x_points[-1] + vx * dt)
        y_points.append(y_points[-1] + vy * dt)
        t += dt
        t_points.append(t)
        if t > 100:
            break
    return np.array(t_points), np.array(x_points), np.array(y_points)

# ================= Trajectory Calculations =================
if air_resistance:
    t_points, x_points, y_points = compute_projectile_with_air(v0, theta, h0)
    t_flight = t_points[-1]
else:
    t_points = np.linspace(0, t_flight, num=num_points)
    x_points = v0 * np.cos(theta) * t_points
    y_points = h0 + v0 * np.sin(theta) * t_points - 0.5 * g * t_points**2
    y_points = np.maximum(y_points, 0)

# ================= Simulation Variables =================
skip_frames = 4
sleep_time = 0.05   # ✅ تعديل: تأخير واضح للحركة والنتائج
x_user = y_user = vy_user = vx_user = v_total_user = None

# ================= Start Button =================
start_button = st.button("🚀 Start Simulation", use_container_width=True)

# ================= Display Interface =================
col_left, col_right = st.columns([2, 1])

with col_left:
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(bg_color)
    x_max = max(x_points) * 1.1
    y_max = max(y_points) * 1.2
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("The Projectile Motion Trajectory")
    ax.grid(True)
    plot_placeholder = st.empty()

with col_right:
    results_placeholder = st.empty()
    analysis_placeholder = st.empty()
    final_placeholder = st.empty()
    progress_bar = st.progress(0)

# ================= Simulation Loop =================
if start_button:
    frame_placeholder = st.empty()  # ✅ تعديل جديد لتحديث الرسم داخل نفس المساحة
    
    max_steps = len(t_points)
    t_step = t_flight / max_steps

    for i_main in range(max_steps):
        current_time = i_main * t_step
        i = np.argmin(np.abs(t_points - current_time))

        # ------------------ رسم الإطار ------------------
        if i_main % skip_frames == 0 or i_main == max_steps - 1:
            ax.clear()
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Range (m)")
            ax.set_ylabel("Height (m)")
            ax.set_title("The Projectile Motion Trajectory")
            ax.grid(True)
            ax.plot(x_points[:i+1], y_points[:i+1], color=trail_color)
            ax.plot(x_points[i], y_points[i], 'ro', markersize=10)

            # ✅ تعديل: التحديث داخل نفس الإطار
            with frame_placeholder:
                st.pyplot(fig)
                time.sleep(sleep_time)

        # ------------------ تحديث النتائج اللحظية ------------------
        if i < len(t_points) - 1:
            vx = (x_points[i+1] - x_points[i]) / (t_points[i+1] - t_points[i])
            vy = (y_points[i+1] - y_points[i]) / (t_points[i+1] - t_points[i])
            v_total = np.sqrt(vx**2 + vy**2)
        else:
            vx = vy = v_total = 0

        results_placeholder.markdown(f"""
        <div style='background-color:#f7f9f9;padding:15px;border-radius:10px;margin-bottom:10px;font-size:16px;'>
            <h4>📊 Instant Results ({angle:.1f}°)</h4>
            <ul>
                <li>⏱  Time: <b>{current_time:.2f}</b> s</li>
                <li>📍  x: <b>{x_points[i]:.2f}</b> m</li>
                <li>📈 y: <b>{y_points[i]:.2f}</b> m</li>
                <li>💨 Vx: <b>{vx:.2f}</b> m/s</li>
                <li>💨 Vy: <b>{vy:.2f}</b> m/s</li>
                <li>💨 V: <b>{v_total:.2f}</b> m/s</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        progress_bar.progress((i_main + 1) / max_steps)

    # ================= Final Results =================
    t_max_height = v0 * np.sin(theta) / g
    final_range_primary = x_points[-1]
    final_max_height = max(y_points)

    final_placeholder.markdown(f"""
    <div style='background-color:#f4ecf7;padding:15px;border-radius:10px;margin-bottom:10px;font-size:16px;'>
        <h3>🏁 Final Results</h3>
        <ul>
            <li>Flight Time: <b>{t_flight:.2f}</b> s</li>
            <li>Time To Reach Max Height: <b>{t_max_height:.2f}</b> s</li>
            <li>Range: <b>{final_range_primary:.2f}</b> m</li>
            <li>Maximum Height: <b>{final_max_height:.2f}</b> m</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
