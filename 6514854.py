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

# Function to calculate flight time
def calculate_flight_time(v0, theta_rad, h0, g):
    return (v0 * np.sin(theta_rad) + np.sqrt((v0*np.sin(theta_rad))**2 + 2*g*h0)) / g

t_flight = calculate_flight_time(v0, theta, h0, g)

if t_user > t_flight:
    st.warning("⏳ The entered time exceeds the flight time! The maximum possible time for the primary trajectory will be used.")
    t_user = t_flight

num_points = 250  # Changed from 600 to 250

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

# Complementary Angle Trajectory Calculation
if compare_angles and angle != 45:
    theta2 = np.radians(90 - angle)
    t_flight2 = calculate_flight_time(v0, theta2, h0, g)
    
    if air_resistance:
        t_points2, x_points2, y_points2 = compute_projectile_with_air(v0, theta2, h0)
        t_flight2 = t_points2[-1]
    else:
        t_points2 = np.linspace(0, t_flight2, num=num_points)
        x_points2 = v0 * np.cos(theta2) * t_points2
        y_points2 = h0 + v0 * np.sin(theta2) * t_points2 - 0.5 * g * t_points2**2
        y_points2 = np.maximum(y_points2, 0)

    t_max = max(t_flight, t_flight2)
else:
    x_points2 = y_points2 = t_points2 = None
    t_flight2 = 0
    t_max = t_flight

# Compare Air Resistance Trajectories
if compare_with_air and not compare_angles: 
    t_flight_no_air = calculate_flight_time(v0, theta, h0, g)
    t_points_no_air = np.linspace(0, t_flight_no_air, num=num_points)
    x_points_no_air = v0 * np.cos(theta) * t_points_no_air
    y_points_no_air = h0 + v0 * np.sin(theta) * t_points_no_air - 0.5 * g * t_points_no_air**2
    y_points_no_air = np.maximum(y_points_no_air, 0)
    
    if air_resistance:
        t_points_with_air, x_points_with_air, y_points_with_air = t_points, x_points, y_points
    else:
        t_points_with_air, x_points_with_air, y_points_with_air = compute_projectile_with_air(v0, theta, h0)
    
    t_max = max(t_flight, t_points_with_air[-1]) if air_resistance else max(t_flight_no_air, t_points_with_air[-1])
else:
    t_points_no_air = x_points_no_air = y_points_no_air = None
    t_points_with_air = x_points_with_air = y_points_with_air = None

# ================= Simulation Variables Initialization =================
analysis_done = False
skip_frames = 12  # Changed from 8 to 12
sleep_time = 0.002  # Changed from 0.0005 to 0.002
x_user = y_user = vy_user = vx_user = v_total_user = None

# ================= Start Button =================
start_button = st.button("🚀 Start Simulation", use_container_width=True)

# ================= Display Interface =================
col_left, col_right = st.columns([2, 1])

with col_left:
    fig, ax = plt.subplots(figsize=(8, 5))  # Added figsize
    fig.patch.set_facecolor(bg_color)
    
    # Determine max range and height for plotting
    x_max = max(x_points) * 1.1
    y_max = max(y_points) * 1.2
    
    if compare_angles and x_points2 is not None:
        x_max = max(x_max, max(x_points2) * 1.1)
        y_max = max(y_max, max(y_points2) * 1.2)
    
    if compare_with_air and not compare_angles:
        x_max = max(x_max, max(x_points_no_air) * 1.1, max(x_points_with_air) * 1.1)
        y_max = max(y_max, max(y_points_no_air) * 1.2, max(y_points_with_air) * 1.2)
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("The Projectile Motion Trajectory")
    ax.grid(True)
    plot_placeholder = st.empty()

    # ------------------ Interactive Questions System ------------------
    teacher_view, student_view = st.tabs(["👨‍🏫 Teacher Mode", "👩‍🎓 Student Mode"])

    with teacher_view:
        st.markdown("## ✍️ Enter Questions (Not saved after closing)")
        with st.form("add_question_form"):
            q_text = st.text_area("Question Text:")
            options = [st.text_input(f"Option {i+1}") for i in range(4)]
            correct_index = st.selectbox("Correct Option Index (1-4):", [1,2,3,4])
            submitted = st.form_submit_button("➕ Add Question")

            if submitted and q_text and all(options):
                st.session_state.questions.append({
                    "question": q_text,
                    "options": options,
                    "correct_index": correct_index - 1
                })
                st.success(f"✅ Question {len(st.session_state.questions)} added successfully")

        if st.session_state.questions:
            st.write("### 🧾 Questions List:")
            
            for i, q in enumerate(st.session_state.questions, 1):
                col_q, col_btn = st.columns([3, 1])
                with col_q:
                    st.write(f"{i}. {q['question']}")
                with col_btn:
                    if st.button(f"✏️ Edit", key=f"edit_{i-1}"):
                        st.session_state.edit_index = i-1
                        st.session_state.edit_mode = True
                        st.rerun()

            if st.session_state.edit_mode and "edit_index" in st.session_state:
                edit_idx = st.session_state.edit_index
                if edit_idx < len(st.session_state.questions):
                    current_q = st.session_state.questions[edit_idx]

                    with st.form(f"edit_form_{edit_idx}"):
                        st.markdown("### ✏️ Edit Question")
                        edited_q = st.text_area("Question Text:", value=current_q["question"])
                        edited_options = []
                        for j in range(4):
                            edited_options.append(st.text_input(f"Option {j+1}:", 
                                                              value=current_q["options"][j],
                                                              key=f"edit_opt_{j}_{edit_idx}"))
                        edited_correct = st.selectbox("Correct Answer:", [1,2,3,4], 
                                                    index=current_q["correct_index"],
                                                    key=f"edit_correct_{edit_idx}")
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            save_edit = st.form_submit_button("💾 Save Changes")
                        with col_cancel:
                            cancel_edit = st.form_submit_button("❌ Cancel")
                        
                        if save_edit:
                            if edited_q and all(edited_options):
                                st.session_state.questions[edit_idx] = {
                                    "question": edited_q,
                                    "options": edited_options,
                                    "correct_index": edited_correct - 1
                                }
                                st.success("✅ Changes saved successfully!")
                                st.session_state.edit_mode = False
                                del st.session_state.edit_index
                                st.rerun()
                        
                        if cancel_edit:
                            st.session_state.edit_mode = False
                            del st.session_state.edit_index
                            st.rerun()

        if st.button("🗑️ Clear All Questions"):
            st.session_state.questions.clear()
            st.session_state.current_question = 0
            st.session_state.test_completed = False
            st.session_state.show_answer_result = False
            st.session_state.correct_answered = False
            st.session_state.edit_mode = False
            st.success("All questions cleared successfully.")
            st.rerun()

    with student_view:
        st.markdown("## 🧩 Projectile Test")
        if not st.session_state.questions:
            st.warning("Questions are not set up yet. Please wait for the teacher.")
        else:
            if st.session_state.test_completed:
                st.balloons()
                st.success("🎉 Test completed successfully!")
                if st.button("🔄 Restart Test"):
                    st.session_state.current_question = 0
                    st.session_state.test_completed = False
                    st.session_state.show_answer_result = False
                    st.session_state.correct_answered = False
                    st.rerun()
            else:
                idx = st.session_state.current_question
                question = st.session_state.questions[idx]

                st.write(f"### Question {idx+1} of {len(st.session_state.questions)}")
                st.write(f"**{question['question']}**")
                
                choice = st.radio("Choose the answer:", question["options"], key=f"q_{idx}")

                if not st.session_state.correct_answered:
                    if st.button("✅ Check Answer", key=f"check_{idx}"):
                        if question["options"].index(choice) == question["correct_index"]:
                            st.session_state.show_answer_result = "correct"
                            st.session_state.correct_answered = True
                        else:
                            st.session_state.show_answer_result = "wrong"
                        st.rerun()
                
                if st.session_state.show_answer_result:
                    if st.session_state.show_answer_result == "correct":
                        st.success("🌟 Correct Answer!")
                        
                        col1, col2 = st.columns([1, 2])
                        with col2:
                            if st.button("➡️ Next" if idx + 1 < len(st.session_state.questions) else "🏁 Finish Test", 
                                       key=f"next_{idx}"):
                                if idx + 1 < len(st.session_state.questions):
                                    st.session_state.current_question += 1
                                    st.session_state.show_answer_result = False
                                    st.session_state.correct_answered = False
                                    st.rerun()
                                else:
                                    st.session_state.test_completed = True
                                    st.session_state.show_answer_result = False
                                    st.session_state.correct_answered = False
                                    st.rerun()
                    else:
                        st.error("❌ Wrong Answer, try again!")

with col_right:
    results_placeholder = st.empty()
    analysis_placeholder = st.empty()
    final_placeholder = st.empty()
    
    # Hide Progress Bar when comparing complementary angles
    if not compare_angles:
        progress_bar = st.progress(0)
    else:
        progress_bar_placeholder = st.empty()

# ================= Simulation Loop (Independent Trajectories) =================
if start_button:
    
    # Determine max steps based on max time
    if not air_resistance:
        max_steps = num_points
    else:
        max_steps = len(t_points)
        if t_points2 is not None:
             max_steps = max(max_steps, len(t_points2))
        if compare_with_air and t_points_with_air is not None:
             max_steps = max(max_steps, len(t_points_with_air))
    
    t_step = t_max / max_steps
    
    finished_primary = False
    finished_secondary = False
    
    analysis_done = False
    x_user = y_user = vy_user = vx_user = v_total_user = None

    # Main Loop based on Max Time
    for i_main in range(max_steps):
        
        current_time = i_main * t_step 
        
        # ------------------ Primary Trajectory (θ) ------------------
        if not finished_primary:
            if current_time <= t_flight:
                i = np.argmin(np.abs(t_points - current_time))
                if y_points[i] <= 0 and i > 5:
                    finished_primary = True
                    i = np.where(y_points <= 0)[0][0] if np.any(y_points <= 0) else len(t_points) - 1
                is_primary_flying = not finished_primary
            else:
                i = len(t_points) - 1
                is_primary_flying = False
                finished_primary = True
        else:
            i = len(t_points) - 1
            is_primary_flying = False
        
        # ------------------ Secondary Trajectory (90°-θ) ------------------
        if compare_angles and t_points2 is not None:
            if not finished_secondary:
                if current_time <= t_flight2:
                    j = np.argmin(np.abs(t_points2 - current_time))
                    if y_points2[j] <= 0 and j > 5:
                        finished_secondary = True
                        j = np.where(y_points2 <= 0)[0][0] if np.any(y_points2 <= 0) else len(t_points2) - 1
                    is_secondary_flying = not finished_secondary
                else:
                    j = len(t_points2) - 1
                    is_secondary_flying = False
                    finished_secondary = True
            else:
                j = len(t_points2) - 1
                is_secondary_flying = False

        # ------------------ Other Trajectories (Air Comparison) ------------------
        if compare_with_air and not compare_angles:
            k_no_air = np.argmin(np.abs(t_points_no_air - current_time)) if current_time <= t_points_no_air[-1] else len(t_points_no_air) - 1
            k_with_air = np.argmin(np.abs(t_points_with_air - current_time)) if current_time <= t_points_with_air[-1] else len(t_points_with_air) - 1

        # ------------------ Update Plot ------------------
        if i_main % skip_frames == 0 or i_main == max_steps - 1:
            ax.clear()
            
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Range (m)")
            ax.set_ylabel("Height (m)")
            ax.set_title("The Projectile Motion Trajectory")
            ax.grid(True)

            # Plot Primary
            ax.plot(x_points[:i+1], y_points[:i+1], color=trail_color, label=f"{angle:.1f}°")
            if is_primary_flying:
                 ax.plot(x_points[i], y_points[i], 'ro', markersize=10) 

            # Plot Secondary
            if compare_angles and t_points2 is not None:
                ax.plot(x_points2[:j+1], y_points2[:j+1], color='orange', linestyle='--', label=f"{90-angle:.1f}°")
                if is_secondary_flying:
                     ax.plot(x_points2[j], y_points2[j], 'bs', markersize=8)

            # Plot Air Comparison
            if compare_with_air and not compare_angles:
                ax.plot(x_points_no_air[:k_no_air+1], y_points_no_air[:k_no_air+1], color='blue', linestyle='-', label=f"{angle:.1f}° (No Air)")
                ax.plot(x_points_with_air[:k_with_air+1], y_points_with_air[:k_with_air+1], color='red', linestyle='--', label=f"{angle:.1f}° (With Air)")

            ax.legend()
            plot_placeholder.pyplot(fig)
            plt.close(fig)  # Added to prevent memory leaks

        # ------------------ Update Instantaneous Results ------------------
        if is_primary_flying:
            # Calculate instantaneous velocities for the primary trajectory
            i_next = min(i + 1, len(t_points) - 1)
            
            if i_next > i: 
                vx = x_points[i_next] - x_points[i]
                vy = y_points[i_next] - y_points[i]
                dt_step = t_points[i_next] - t_points[i]
                
                if dt_step > 0:
                    vx_instant = vx / dt_step
                    vy_instant = vy / dt_step
                    v_total = np.sqrt(vx_instant**2 + vy_instant**2)
                else:
                    vx_instant = vy_instant = v_total = 0.0
            else:
                vx_instant = vy_instant = v_total = 0.0

            results_placeholder.markdown(f"""
            <div style='background-color:#f7f9f9;padding:15px;border-radius:10px;margin-bottom:10px;font-size:16px;'>
                <h4>📊 Instant Results ({angle:.1f}°)</h4>
                <ul>
                    <li>⏱  Time: <b>{current_time:.2f}</b> (s)</li>
                    <li>📍  Horizontal Distance (x): <b>{x_points[i]:.2f}</b> m</li>
                    <li>📈 Height (y): <b>{y_points[i]:.2f}</b> m</li>
                    <li>💨 Vertical Velocity (Vy): <b>{vy_instant:.2f}</b> m/s</li>
                    <li>💨 Horizontal Velocity (Vx): <b>{vx_instant:.2f}</b> m/s</li>
                    <li>💨 Net Velocity (V): <b>{v_total:.2f}</b> m/s</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if not analysis_done and t_user > 0 and current_time >= t_user:
                x_user = x_points[i]
                y_user = y_points[i]
                vy_user = vy_instant
                vx_user = vx_instant
                v_total_user = v_total
                analysis_placeholder.markdown(f"""
                <div style='background-color:#eaf2f8;padding:15px;border-radius:10px;margin-bottom:10px;font-size:16px;'>
                    <h4>🔍 Analysis At t = {t_user:.2f} s</h4>
                    <ul>
                        <li>📍  Range: <b>{x_user:.2f}</b> m</li>
                        <li>📈 Height: <b>{y_user:.2f}</b> m</li>
                        <li>💨 Vertical Velocity: <b>{vy_user:.2f}</b> m\s </li>
                        <li>💨 Horizontal Velocity: <b>{vx_user:.2f}</b> m\s </li>
                        <li>💨 Net Velocity: <b>{v_total_user:.2f}</b> m\s </li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                analysis_done = True
        else:
            results_placeholder.empty()

        # ------------------ Update Progress Bar ------------------
        if not compare_angles:
            progress_bar.progress((i_main + 1) / max_steps)
            
        # ------------------ Stop Condition ------------------
        stop_condition = finished_primary
        if compare_angles:
            stop_condition = stop_condition and finished_secondary

        if stop_condition or (compare_with_air and not compare_angles and current_time >= t_max):
            break

        time.sleep(sleep_time)

    # Finalize Progress Bar
    if not compare_angles:
        progress_bar.progress(100)
    
    results_placeholder.empty()
    
    # Final Results Calculations
    t_max_height = v0 * np.sin(theta) / g
    
    # We use the final index (i) reached by the primary trajectory
    final_range_primary = x_points[i]
    final_max_height = max(y_points)

    # ================= FINAL RESULTS =================
    final_placeholder.markdown(f"""
    <div style='background-color:#f4ecf7;padding:15px;border-radius:10px;margin-bottom:10px;font-size:16px;'>
        <h3>🏁 Final Results</h3>
        <ul>
            <li>Flight Time (t'): <b>{t_flight:.2f}</b> s</li>
            <li>Time To Reach Maximum Height: <b>{t_max_height:.2f}</b> s</li>
            <li>Range: <b>{final_range_primary:.2f}</b> m</li>
            <li>Maximum Height: <b>{final_max_height:.2f}</b> m </li>
            <li>Trajectory Equation: <b>y(x) = {h0:.2f} + x.tan({angle:.2f}°) - (9.81/(2*({v0:.2f}·cos({angle:.2f}°))²)).x²</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
