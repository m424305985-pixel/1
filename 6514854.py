import streamlit as st
import numpy as np
import plotly.graph_objects as go

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

# ================= Physics and Helper Functions =================
g = 9.81

def calculate_flight_time(v0, theta_rad, h0, g=9.81):
    # closed-form for projectile (no air)
    disc = (v0*np.sin(theta_rad))**2 + 2*g*h0
    return (v0 * np.sin(theta_rad) + np.sqrt(disc)) / g

def compute_projectile_with_air(v0, theta, h0, g=9.81, c=0.005, dt=0.01, t_max=100.0):
    # simple drag proportional to v (linear-ish), small dt integration
    t_points = [0.0]
    x_points = [0.0]
    y_points = [h0]
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    t = 0.0
    while y_points[-1] >= 0 and t < t_max:
        v = np.sqrt(vx*vx + vy*vy)
        ax = -c * v * vx
        ay = -g - c * v * vy
        vx = vx + ax * dt
        vy = vy + ay * dt
        x_points.append(x_points[-1] + vx * dt)
        y_points.append(y_points[-1] + vy * dt)
        t += dt
        t_points.append(t)
    return np.array(t_points), np.array(x_points), np.array(y_points)

# ================= Prepare Trajectories Based on Options =================
theta = np.radians(angle)

# Primary trajectory (may use air resistance)
if air_resistance:
    t_p, x_p, y_p = compute_projectile_with_air(v0, theta, h0)
    t_flight_primary = t_p[-1]
else:
    t_flight_primary = calculate_flight_time(v0, theta, h0, g)
    # choose a reasonable number of frames
    t_p = np.linspace(0, t_flight_primary, 180)
    x_p = v0 * np.cos(theta) * t_p
    y_p = h0 + v0 * np.sin(theta) * t_p - 0.5 * g * (t_p**2)
    y_p = np.maximum(y_p, 0)

# Secondary complementary angle (if requested)
if compare_angles and angle != 45:
    theta2 = np.radians(90 - angle)
    if air_resistance:
        t_s, x_s, y_s = compute_projectile_with_air(v0, theta2, h0)
        t_flight_secondary = t_s[-1]
    else:
        t_flight_secondary = calculate_flight_time(v0, theta2, h0, g)
        t_s = np.linspace(0, t_flight_secondary, 180)
        x_s = v0 * np.cos(theta2) * t_s
        y_s = h0 + v0 * np.sin(theta2) * t_s - 0.5 * g * (t_s**2)
        y_s = np.maximum(y_s, 0)
else:
    t_s = np.array([])
    x_s = np.array([])
    y_s = np.array([])

# Compare with/without air
if compare_with_air and not compare_angles:
    # compute no-air and with-air for comparison
    t_no_air = np.linspace(0, calculate_flight_time(v0, theta, h0, g), 180)
    x_no_air = v0 * np.cos(theta) * t_no_air
    y_no_air = h0 + v0 * np.sin(theta) * t_no_air - 0.5 * g * (t_no_air**2)
    y_no_air = np.maximum(y_no_air, 0)

    if air_resistance:
        # with air already computed as t_p, x_p, y_p
        t_with_air, x_with_air, y_with_air = t_p, x_p, y_p
    else:
        t_with_air, x_with_air, y_with_air = compute_projectile_with_air(v0, theta, h0)
else:
    t_no_air = np.array([])
    x_no_air = np.array([])
    y_no_air = np.array([])
    t_with_air = np.array([])
    x_with_air = np.array([])
    y_with_air = np.array([])

# ================= UI Columns and Placeholders =================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 🎮 Simulation View")
    plot_placeholder = st.empty()

with col_right:
    results_placeholder = st.empty()
    analysis_placeholder = st.empty()
    final_placeholder = st.empty()

# ================= Teacher/Student Questions Section (kept intact) =================
# (This block is kept exactly as you had it — only minor key names kept consistent)
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

# ================= Simulation / Plotly Animation Building =================

# Determine max extents for plot ranges
x_candidates = [x_p] 
y_candidates = [y_p]
if x_s.size:
    x_candidates.append(x_s)
    y_candidates.append(y_s)
if x_no_air.size:
    x_candidates.append(x_no_air)
    y_candidates.append(y_no_air)
if x_with_air.size:
    x_candidates.append(x_with_air)
    y_candidates.append(y_with_air)

# ensure non-empty
x_max = max([arr.max() if arr.size else 0 for arr in x_candidates]) * 1.1
y_max = max([arr.max() if arr.size else 0 for arr in y_candidates]) * 1.2
x_max = max(x_max, 1.0)
y_max = max(y_max, 1.0)

# Build frame-aligned arrays (pad shorter arrays by repeating last point)
def pad_to_length(arr, n):
    if arr.size == 0:
        return np.full(n, np.nan)
    if len(arr) >= n:
        return arr[:n]
    last = arr[-1]
    return np.concatenate([arr, np.full(n - len(arr), last)])

# choose frame count as max length across trajectories
lengths = [len(t_p)]
if len(t_s) > 0:
    lengths.append(len(t_s))
if len(t_no_air) > 0:
    lengths.append(len(t_no_air))
if len(t_with_air) > 0:
    lengths.append(len(t_with_air))
max_frames = max(lengths)

# build padded arrays for frames
x_p_pad = pad_to_length(x_p, max_frames)
y_p_pad = pad_to_length(y_p, max_frames)
t_p_pad = pad_to_length(t_p, max_frames)

x_s_pad = pad_to_length(x_s, max_frames)
y_s_pad = pad_to_length(y_s, max_frames)
t_s_pad = pad_to_length(t_s, max_frames)

x_no_air_pad = pad_to_length(x_no_air, max_frames)
y_no_air_pad = pad_to_length(y_no_air, max_frames)
t_no_air_pad = pad_to_length(t_no_air, max_frames)

x_with_air_pad = pad_to_length(x_with_air, max_frames)
y_with_air_pad = pad_to_length(y_with_air, max_frames)
t_with_air_pad = pad_to_length(t_with_air, max_frames)

# Precompute instantaneous data for primary (and secondary if present)
inst_primary = []
for i in range(max_frames):
    if i < max_frames - 1:
        dt = (t_p_pad[i+1] - t_p_pad[i]) if not np.isnan(t_p_pad[i+1]) and not np.isnan(t_p_pad[i]) else 1e-6
        vx = (x_p_pad[i+1] - x_p_pad[i]) / dt if not np.isnan(x_p_pad[i]) else 0.0
        vy = (y_p_pad[i+1] - y_p_pad[i]) / dt if not np.isnan(y_p_pad[i]) else 0.0
    else:
        vx = vy = 0.0
    v = np.sqrt(vx*vx + vy*vy)
    inst_primary.append({
        "t": float(t_p_pad[i]) if not np.isnan(t_p_pad[i]) else 0.0,
        "x": float(x_p_pad[i]) if not np.isnan(x_p_pad[i]) else 0.0,
        "y": float(y_p_pad[i]) if not np.isnan(y_p_pad[i]) else 0.0,
        "vx": float(vx),
        "vy": float(vy),
        "v": float(v)
    })

inst_secondary = []
if x_s_pad.size and not np.all(np.isnan(x_s_pad)):
    for i in range(max_frames):
        if i < max_frames - 1:
            dt = (t_s_pad[i+1] - t_s_pad[i]) if not np.isnan(t_s_pad[i+1]) and not np.isnan(t_s_pad[i]) else 1e-6
            vx = (x_s_pad[i+1] - x_s_pad[i]) / dt if not np.isnan(x_s_pad[i]) else 0.0
            vy = (y_s_pad[i+1] - y_s_pad[i]) / dt if not np.isnan(y_s_pad[i]) else 0.0
        else:
            vx = vy = 0.0
        v = np.sqrt(vx*vx + vy*vy)
        inst_secondary.append({
            "t": float(t_s_pad[i]) if not np.isnan(t_s_pad[i]) else 0.0,
            "x": float(x_s_pad[i]) if not np.isnan(x_s_pad[i]) else 0.0,
            "y": float(y_s_pad[i]) if not np.isnan(y_s_pad[i]) else 0.0,
            "vx": float(vx),
            "vy": float(vy),
            "v": float(v)
        })

# Build Plotly figure with frames
fig = go.Figure()

# Static trajectory traces
fig.add_trace(go.Scatter(x=x_p, y=y_p, mode='lines', name=f"{angle:.1f}° (Primary)", line=dict(color='black')))
if x_s.size:
    fig.add_trace(go.Scatter(x=x_s, y=y_s, mode='lines', name=f"{90-angle:.1f}° (Secondary)", line=dict(color='orange', dash='dash')))
if x_no_air.size:
    fig.add_trace(go.Scatter(x=x_no_air, y=y_no_air, mode='lines', name=f"{angle:.1f}° (No Air)", line=dict(color='blue')))
if x_with_air.size:
    fig.add_trace(go.Scatter(x=x_with_air, y=y_with_air, mode='lines', name=f"{angle:.1f}° (With Air)", line=dict(color='red', dash='dot')))

# Moving markers (one per active trajectory)
# Primary marker
fig.add_trace(go.Scatter(x=[x_p_pad[0]], y=[y_p_pad[0]], mode='markers', marker=dict(size=12, color='red'), name="Ball (Primary)"))
# Secondary marker
if x_s.size:
    fig.add_trace(go.Scatter(x=[x_s_pad[0]], y=[y_s_pad[0]], mode='markers', marker=dict(size=10, color='blue'), name="Ball (Secondary)"))
# No-air / with-air markers if applicable
if x_no_air.size and x_with_air.size:
    fig.add_trace(go.Scatter(x=[x_no_air_pad[0]], y=[y_no_air_pad[0]], mode='markers', marker=dict(size=8, color='cyan'), name="Ball (No Air)"))
    fig.add_trace(go.Scatter(x=[x_with_air_pad[0]], y=[y_with_air_pad[0]], mode='markers', marker=dict(size=8, color='magenta'), name="Ball (With Air)"))

# Frames: update positions of moving markers
frames = []
for frame_idx in range(max_frames):
    data = []
    # keep static trajectory lines as-is (Plotly's frames can omit them)
    # moving markers: must match order of traces added
    # first moving marker: primary (trace index = number of static traces)
    # Count number of static traces to compute marker trace indices
    static_count = 0
    static_count += 1  # primary line
    if x_s.size:
        static_count += 1
    if x_no_air.size:
        static_count += 1
    if x_with_air.size:
        static_count += 1

    # Build data updates in the same order as moving marker traces were appended
    # Primary marker
    px = x_p_pad[frame_idx] if not np.isnan(x_p_pad[frame_idx]) else None
    py = y_p_pad[frame_idx] if not np.isnan(y_p_pad[frame_idx]) else None
    data.append(go.Scatter(x=[px], y=[py], mode='markers', marker=dict(size=12, color='red')))

    # Secondary marker (if exists)
    if x_s.size:
        sx = x_s_pad[frame_idx] if not np.isnan(x_s_pad[frame_idx]) else None
        sy = y_s_pad[frame_idx] if not np.isnan(y_s_pad[frame_idx]) else None
        data.append(go.Scatter(x=[sx], y=[sy], mode='markers', marker=dict(size=10, color='blue')))

    # No-air / with-air markers (if exist)
    if x_no_air.size and x_with_air.size:
        nax = x_no_air_pad[frame_idx] if not np.isnan(x_no_air_pad[frame_idx]) else None
        nay = y_no_air_pad[frame_idx] if not np.isnan(y_no_air_pad[frame_idx]) else None
        wax = x_with_air_pad[frame_idx] if not np.isnan(x_with_air_pad[frame_idx]) else None
        way = y_with_air_pad[frame_idx] if not np.isnan(y_with_air_pad[frame_idx]) else None
        data.append(go.Scatter(x=[nax], y=[nay], mode='markers', marker=dict(size=8, color='cyan')))
        data.append(go.Scatter(x=[wax], y=[way], mode='markers', marker=dict(size=8, color='magenta')))

    frames.append(go.Frame(data=data, name=str(frame_idx)))

fig.frames = frames

# Layout: add play/pause buttons and a slider
fig.update_layout(
    xaxis=dict(range=[0, x_max], title="Range (m)"),
    yaxis=dict(range=[0, y_max], title="Height (m)"),
    title="Projectile Motion Animation (Plotly)",
    updatemenus=[{
        "buttons": [
            {"args": [None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}], "label": "▶️ Play", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": "⏸ Pause", "method": "animate"}
        ],
        "direction": "left", "pad": {"r": 10, "t": 70}, "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "currentvalue": {"prefix": "Frame: "},
        "pad": {"t": 50},
        "steps": [{"args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(k), "method": "animate"} for k in range(max_frames)]
    }]
)

# Display the Plotly chart
plot_placeholder.plotly_chart(fig, use_container_width=True)

# ================= Instantaneous Results Display & Controls =================
st.markdown("---")
st.markdown("### 📊 Instantaneous Results")

# Note to user about Plotly client-side animation
st.info("ملاحظة: عند الضغط على Play داخل الرسم، التحريك يتم في المتصفح (client-side) ولذلك نتائج Streamlit (التي تعمل على الخادم) لن تتغير تلقائيًا أثناء التشغيل. استخدم شريط الإطار (Frame slider) أو أزرار Play/Pause في الرسم لمزامنة الإطار الذي تعرضه، ثم استعمل شريط الإطار أدناه لمشاهدة النتائج المقابلة.")

# Frame selector (Streamlit-side) to show results at a chosen frame index
frame_idx = st.slider("Select frame index to show instantaneous results", min_value=0, max_value=max_frames-1, value=0, step=1)

# Display primary instantaneous results
prim = inst_primary[frame_idx]
results_placeholder.markdown(f"""
<div style='background-color:#f7f9f9;padding:12px;border-radius:8px;'>
  <h4>Primary ({angle:.1f}°)</h4>
  <ul>
    <li>⏱ Time: <b>{prim['t']:.3f} s</b></li>
    <li>📍 x: <b>{prim['x']:.3f} m</b></li>
    <li>📈 y: <b>{prim['y']:.3f} m</b></li>
    <li>💨 Vx: <b>{prim['vx']:.3f} m/s</b></li>
    <li>💨 Vy: <b>{prim['vy']:.3f} m/s</b></li>
    <li>💨 V: <b>{prim['v']:.3f} m/s</b></li>
  </ul>
</div>
""", unsafe_allow_html=True)

# If secondary exists, display it as well
if inst_secondary:
    sec = inst_secondary[frame_idx]
    analysis_placeholder.markdown(f"""
    <div style='background-color:#eef6ff;padding:12px;border-radius:8px;'>
      <h4>Secondary ({90-angle:.1f}°)</h4>
      <ul>
        <li>⏱ Time: <b>{sec['t']:.3f} s</b></li>
        <li>📍 x: <b>{sec['x']:.3f} m</b></li>
        <li>📈 y: <b>{sec['y']:.3f} m</b></li>
        <li>💨 Vx: <b>{sec['vx']:.3f} m/s</b></li>
        <li>💨 Vy: <b>{sec['vy']:.3f} m/s</b></li>
        <li>💨 V: <b>{sec['v']:.3f} m/s</b></li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# Final results block (keeps original style)
t_max_height = v0 * np.sin(theta) / g
final_range_primary = x_p[-1] if x_p.size else 0.0
final_max_height = max(y_p) if y_p.size else 0.0

final_placeholder.markdown(f"""
<div style='background-color:#f4ecf7;padding:15px;border-radius:10px;margin-bottom:10px;font-size:16px;'>
    <h3>🏁 Final Results</h3>
    <ul>
        <li>Flight Time: <b>{t_flight_primary:.2f}</b> s</li>
        <li>Time To Reach Max Height: <b>{t_max_height:.2f}</b> s</li>
        <li>Range: <b>{final_range_primary:.2f}</b> m</li>
        <li>Maximum Height: <b>{final_max_height:.2f}</b> m </li>
        <li>Trajectory Equation: <b>y(x) = {h0:.2f} + x·tan({angle:.2f}°) - (9.81/(2*({v0:.2f}·cos({angle:.2f}°))²)).x²</b></li>
    </ul>
</div>
""", unsafe_allow_html=True)
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

num_points = 600

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
skip_frames = 8
sleep_time = 0.0005
x_user = y_user = vy_user = vx_user = v_total_user = None

# ================= Start Button =================
start_button = st.button("🚀 Start Simulation", use_container_width=True)


# ================= Display Interface =================
col_left, col_right = st.columns([2, 1])

with col_left:
    fig, ax = plt.subplots()
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
            if analysis_done:
                ax.plot(x_user, y_user, 'gs', markersize=10, label=f"Analysis at {t_user:.2f}s")
                ax.legend()
            plot_placeholder.pyplot(fig)

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

    # ================= RESTORED ORIGINAL FINAL RESULTS BLOCK (SIMPLE) =================
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

