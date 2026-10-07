import time
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# 1. 頁面配置 (Page Configuration)
st.set_page_config(
    page_title="🎲 自訂骰子相配實驗 | Multi-Dice Experiment",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS 視覺化樣式
custom_css = textwrap.dedent("""
<style>
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #ffffff !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif !important;
    color: #1c1c1e !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px;
}

.card {
    background: #ffffff;
    border-radius: 20px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    border: 1px solid rgba(0, 0, 0, 0.06);
}

.ios-kpi-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 14px 10px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
    border: 1px solid rgba(0, 0, 0, 0.06);
}

.ios-kpi-title {
    font-size: 12px;
    font-weight: 500;
    color: #8e8e93;
    margin-bottom: 4px;
    line-height: 1.3;
}

.ios-kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #1c1c1e;
}

.dice-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    margin: 20px 0;
    padding: 10px 0;
    flex-wrap: wrap;
}

.dice-card {
    min-width: 75px;
    height: 85px;
    padding: 8px;
    background: #ffffff;
    border-radius: 18px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.08);
    transition: transform 0.3s ease;
}

.dice-icon {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 2px;
    color: #1c1c1e;
}

.dice-label {
    font-size: 10px;
    font-weight: 600;
    color: #8e8e93;
    text-align: center;
    line-height: 1.1;
}

.dice-card.matched {
    background: linear-gradient(135deg, #34c759 0%, #28a745 100%);
    border: none;
    box-shadow: 0 10px 22px rgba(52, 199, 89, 0.4);
    transform: scale(1.1);
    z-index: 10;
}

.dice-card.matched .dice-icon,
.dice-card.matched .dice-label {
    color: #ffffff !important;
}

div[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
    border-right: 1px solid rgba(0, 0, 0, 0.06);
}

div.stButton > button {
    border-radius: 16px !important;
    background-color: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    padding: 0.6rem 0.25rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

div.stButton > button, div.stButton > button p, div.stButton > button span {
    color: #1c1c1e !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

div.stButton > button:hover {
    background-color: #f2f2f7 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08) !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #007aff 0%, #0056b3 100%) !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3) !important;
}

div.stButton > button[kind="primary"], 
div.stButton > button[kind="primary"] p, 
div.stButton > button[kind="primary"] span {
    color: #ffffff !important;
}
</style>
""")

st.markdown(custom_css, unsafe_allow_html=True)

# 3. 輔助工具：多骰子機率計算 (Convolution)
def compute_exact_pmf(num_dice, num_sides):
    single_die = np.ones(num_sides) / num_sides
    pmf = single_die
    for _ in range(num_dice - 1):
        pmf = np.convolve(pmf, single_die)
    
    x_values = np.arange(num_dice, num_dice * num_sides + 1)
    total_outcomes = num_sides ** num_dice
    ways = np.round(pmf * total_outcomes).astype(int)
    return x_values, ways, pmf, total_outcomes

DICE_ICONS = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6'
}

# 4. Session State 初始化與同步
if "anim_status" not in st.session_state:
    st.session_state.anim_status = "idle"
if "current_step_idx" not in st.session_state:
    st.session_state.current_step_idx = 0
if "total_n" not in st.session_state:
    st.session_state.total_n = 1000
if "slider_n" not in st.session_state:
    st.session_state.slider_n = 1000
if "input_n" not in st.session_state:
    st.session_state.input_n = 1000

def sync_from_slider():
    val = st.session_state.slider_n
    st.session_state.total_n = val
    st.session_state.input_n = val
    st.session_state.anim_status = "idle"
    st.session_state.current_step_idx = 0

def sync_from_input():
    val = st.session_state.input_n
    st.session_state.total_n = val
    st.session_state.slider_n = val
    st.session_state.anim_status = "idle"
    st.session_state.current_step_idx = 0

def reset_anim():
    st.session_state.anim_status = "idle"
    st.session_state.current_step_idx = 0

# 5. 側邊欄 (Sidebar)
with st.sidebar:
    st.header("⚙️ 控制面板 (Control Panel)")
    
    st.markdown("**🎲 骰子參數設定**")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        num_dice = st.number_input("骰子數量", min_value=1, max_value=10, value=2, step=1, on_change=reset_anim)
    with col_d2:
        num_sides = st.number_input("骰子面數", min_value=2, max_value=100, value=6, step=1, on_change=reset_anim)

    st.markdown("**模擬總次數 (N) / Total Simulations (N)**")
    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        st.slider("拉動次數", 100, 10000, step=100, key="slider_n", on_change=sync_from_slider, label_visibility="collapsed")
    with col_s2:
        st.number_input("輸入次數", 100, 10000, step=100, key="input_n", on_change=sync_from_input, label_visibility="collapsed")

    total_n = st.session_state.total_n
    fps = st.slider("動畫速率 (FPS) / Speed", 2, 25, 8)
    seed = st.number_input("隨機種子 (Seed) / Random Seed", 0, 9999, 42, on_change=reset_anim)

    st.divider()

    col_b1, col_b2, col_b3 = st.columns(3)
    start_click = col_b1.button("🚀 開始", type="primary", use_container_width=True)
    pause_label = "▶️ 繼續" if st.session_state.anim_status == "paused" else "⏸️ 暫停"
    pause_click = col_b2.button(pause_label, use_container_width=True)
    quick_click = col_b3.button("⚡ 結算", use_container_width=True)

    if start_click:
        st.session_state.anim_status = "running"
        st.session_state.current_step_idx = 0
        st.rerun()

    if pause_click:
        if st.session_state.anim_status == "running":
            st.session_state.anim_status = "paused"
        elif st.session_state.anim_status == "paused":
            st.session_state.anim_status = "running"
        st.rerun()

    if quick_click:
        st.session_state.anim_status = "finished"
        st.rerun()

# 動態計算當前骰子設定的理論 PMF
X_VALUES, WAYS, THEORETICAL_PMF, TOTAL_OUTCOMES = compute_exact_pmf(num_dice, num_sides)

# 計算理論出現機率最高的點數和 (Mode Sum) 進行重點追蹤
target_x = int(np.floor(num_dice * (num_sides + 1) / 2))
target_x_idx = np.where(X_VALUES == target_x)[0][0]
theory_p_target = THEORETICAL_PMF[target_x_idx]

# 6. 模擬邏輯與 UI 渲染函數
@st.cache_data
def run_simulation(total_n, num_dice, num_sides, seed):
    rng = np.random.default_rng(seed)
    rolls = rng.integers(1, num_sides + 1, size=(total_n, num_dice))
    X = rolls.sum(axis=1)
    
    min_sum = num_dice
    max_sum = num_dice * num_sides
    num_outcomes = max_sum - min_sum + 1
    
    cumulative_counts = np.zeros((total_n, num_outcomes), dtype=int)
    running_counts = np.zeros(num_outcomes, dtype=int)
    
    for i, x in enumerate(X):
        running_counts[x - min_sum] += 1
        cumulative_counts[i] = running_counts
        
    cumulative_relative = cumulative_counts / np.arange(1, total_n + 1)[:, None]
    
    t_idx = target_x - min_sum
    cum_target = cumulative_counts[:, t_idx]
    cum_p_target = cumulative_relative[:, t_idx]
    
    return rolls, X, cumulative_counts, cumulative_relative, cum_target, cum_p_target

def render_dice(single_roll):
    """用真正的骰子點數（pips）顯示每顆骰子，不顯示阿拉伯數字。"""
    total = int(np.sum(single_roll))

    # 每個數字對應骰面上的點位
    pip_positions = {
        1: [(2, 2)],
        2: [(1, 1), (3, 3)],
        3: [(1, 1), (2, 2), (3, 3)],
        4: [(1, 1), (1, 3), (3, 1), (3, 3)],
        5: [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)],
        6: [(1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (3, 3)],
    }

    def die_html(value, matched=False):
        dots = []
        for row, col in pip_positions[int(value)]:
            dots.append(
                f'<span style="grid-row:{row};grid-column:{col};'
                'width:11px;height:11px;border-radius:50%;'
                'background:#1c1c1e;display:block;"></span>'
            )

        border = "#34c759" if matched else "#1c1c1e"
        shadow = (
            "0 0 0 3px rgba(52,199,89,0.18), 0 6px 16px rgba(52,199,89,0.28)"
            if matched else
            "0 4px 14px rgba(0,0,0,0.08)"
        )

        return f"""
        <div style="
            width:76px;
            min-width:76px;
            height:76px;
            padding:10px;
            box-sizing:border-box;
            background:#ffffff;
            border:2px solid {border};
            border-radius:16px;
            display:flex;
            align-items:center;
            justify-content:center;
            box-shadow:{shadow};
        ">
            <div style="
                width:52px;
                height:52px;
                display:grid;
                grid-template-columns:repeat(3, 1fr);
                grid-template-rows:repeat(3, 1fr);
                align-items:center;
                justify-items:center;
            ">
                {''.join(dots)}
            </div>
        </div>
        """

    parts = []
    for idx, val in enumerate(single_roll):
        parts.append(f"""
        <div style="
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            gap:5px;
        ">
            {die_html(int(val))}
            <div style="
                font-size:10px;
                font-weight:600;
                color:#8e8e93;
                white-space:nowrap;
            ">第 {idx + 1} 顆</div>
        </div>
        """)

        if idx < len(single_roll) - 1:
            parts.append("""
            <div style="
                font-size:24px;
                font-weight:700;
                color:#8e8e93;
                padding:0 5px;
            ">+</div>
            """)

    parts.append(f"""
    <div style="
        width:82px;
        min-width:82px;
        height:86px;
        padding:8px;
        box-sizing:border-box;
        background:linear-gradient(135deg,#34c759 0%,#28a745 100%);
        border-radius:18px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        box-shadow:0 10px 22px rgba(52,199,89,0.4);
        text-align:center;
        color:#ffffff;
    ">
        <div style="font-size:13px;font-weight:800;">點數和</div>
        <div style="
            font-size:28px;
            font-weight:700;
            line-height:1.1;
            margin:3px 0;
        ">{total}</div>
        <div style="font-size:10px;font-weight:600;">X = {total}</div>
    </div>
    """)

    html = f"""
    <div style="
        display:flex;
        align-items:center;
        justify-content:center;
        gap:8px;
        width:100%;
        padding:8px 0 12px 0;
        overflow-x:auto;
    ">
        {''.join(parts)}
    </div>
    """

    # st.html 會直接把 HTML 當成網頁內容渲染，不會把標籤顯示成文字。
    st.html(html)

def render_frame_ui(frame_n):
    idx = frame_n - 1
    current_x = int(X[idx])

    spot_kpi1.markdown(render_kpi_html("模擬次數", "Simulations", f"{frame_n}"), unsafe_allow_html=True)
    spot_kpi2.markdown(render_kpi_html("目前 X", "Current X", f"{current_x}", "#007AFF"), unsafe_allow_html=True)
    spot_kpi3.markdown(render_kpi_html(f"估算 P(X={target_x})", f"Estimated P(X={target_x})", f"{cum_p_target[idx]:.4f}", "#007AFF"), unsafe_allow_html=True)
    spot_kpi4.markdown(render_kpi_html(f"理論 P(X={target_x})", f"Theory P(X={target_x})", f"{theory_p_target:.4f}", "#FF3B30"), unsafe_allow_html=True)

    with dice_spot.container():
        render_dice(rolls[idx])
    chart_spot.plotly_chart(
        build_clean_plotly_chart(df_chart.iloc[:frame_n], total_n, target_x),
        use_container_width=True,
        config=plotly_config
    )

# 8. 動畫與狀態控制
if st.session_state.anim_status == "idle":
    render_frame_ui(1)
    st.info("👈 請點擊左側面板的 **「🚀 開始」** 播放動畫，或 **「⚡ 結算」** 直接觀看結果！")

elif st.session_state.anim_status == "finished":
    render_frame_ui(total_n)
    final_error = abs(cum_p_target[-1] - theory_p_target)
    st.success(f"🎉 模擬完成！最終估算 P(X={target_x}) = {cum_p_target[-1]:.4f}，與理論值誤差僅 {final_error:.4f}")

elif st.session_state.anim_status == "paused":
    current_n = int(frame_indices[st.session_state.current_step_idx])
    render_frame_ui(current_n)
    st.warning(f"⏸️ 動畫已暫停於第 {current_n} 次模擬，點擊左側 **「▶️ 繼續」** 可繼續播放。")

elif st.session_state.anim_status == "running":
    progress_bar = st.progress(0)
    start_idx = st.session_state.current_step_idx

    for idx in range(start_idx, len(frame_indices)):
        st.session_state.current_step_idx = idx
        current_n = int(frame_indices[idx])
        progress_bar.progress(int((idx + 1) / len(frame_indices) * 100))
        render_frame_ui(current_n)
        time.sleep(1 / fps)

    progress_bar.empty()
    st.session_state.anim_status = "finished"
    st.rerun()

# 9. 詳細統計結果與圖表
if st.session_state.anim_status in ["paused", "finished"]:
    st.header("📊 (b) 模擬結果與理論機率比較")

    result_df = pd.DataFrame({
        'X (點數和)': X_VALUES,
        '出現次數 / Frequency': final_counts,
        '相對次數 / Relative Frequency': final_relative,
        '理論機率 / Theoretical P(X=x)': THEORETICAL_PMF,
        '誤差 / Error': final_relative - THEORETICAL_PMF
    })

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "相對次數 / Relative Frequency": st.column_config.NumberColumn(format="%.4f"),
            "理論機率 / Theoretical P(X=x)": st.column_config.NumberColumn(format="%.4f"),
            "誤差 / Error": st.column_config.NumberColumn(format="%+.4f")
        }
    )

    st.header("📈 (c) 機率分布對比圖")

    distribution_fig = build_distribution_chart(final_relative)
    st.plotly_chart(distribution_fig, use_container_width=True, config=plotly_config)

    st.subheader("🔎 前 20 次模擬投擲紀錄")
    show_n = min(20, total_n)
    
    first_20_dict = {'第幾次 / Trial': np.arange(1, show_n + 1)}
    for d_i in range(num_dice):
        first_20_dict[f'第 {d_i+1} 顆骰子'] = rolls[:show_n, d_i]
    first_20_dict['X = 點數和'] = X[:show_n]

    st.dataframe(pd.DataFrame(first_20_dict), use_container_width=True, hide_index=True)

    max_error = np.max(np.abs(final_relative - THEORETICAL_PMF))
    mean_error = np.mean(np.abs(final_relative - THEORETICAL_PMF))

    st.subheader("📌 理論值與模擬值總體誤差")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.metric("最大絕對誤差", f"{max_error:.4f}")
    with e2:
        st.metric("平均絕對誤差", f"{mean_error:.4f}")
    with e3:
        st.metric(f"P(X={target_x}) 模擬值", f"{final_relative[target_x_idx]:.4f}")
