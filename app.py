import time
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# 1. 頁面配置 (Page Configuration)
st.set_page_config(
    page_title="🎲 骰子相配實驗 | Dice Matching Experiment",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS style (骰子與按鈕放大效果 Dice and button magnification effect)
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

/* --- 骰子容器 (Dice container) --- */
.dice-wrapper {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 14px;
    margin: 24px 0;
    padding: 15px 0;
    flex-wrap: wrap;
}

.dice-card {
    width: 85px;
    height: 92px;
    background: #ffffff;
    border-radius: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.08);
    
    transform-origin: bottom center;
    transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.35s ease,
                background-color 0.3s ease;
    cursor: pointer;
}

.dice-icon {
    font-size: 34px;
    line-height: 1;
    margin-bottom: 4px;
    color: #1c1c1e;
}

.dice-label {
    font-size: 10px;
    font-weight: 600;
    color: #8e8e93;
    text-align: center;
    line-height: 1.2;
}

.dice-card.matched {
    background: linear-gradient(135deg, #34c759 0%, #28a745 100%);
    border: none;
    box-shadow: 0 14px 28px rgba(52, 199, 89, 0.45);
    transform: translateY(-12px) scale(1.28);
    z-index: 10;
}

.dice-card.matched .dice-icon,
.dice-card.matched .dice-label {
    color: #ffffff !important;
}

.dice-card:hover {
    transform: translateY(-16px) scale(1.35) !important;
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.15) !important;
    z-index: 30 !important;
}

.dice-card:has(+ .dice-card:hover),
.dice-card:hover + .dice-card {
    transform: translateY(-6px) scale(1.12);
    z-index: 20;
}

div[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
    border-right: 1px solid rgba(0, 0, 0, 0.06);
}

/* --- 按鈕風格動畫 (控制面板) Button-style animation (Control Panel) --- */
div.stButton > button {
    border-radius: 16px !important;
    background-color: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    padding: 0.6rem 0.25rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    
    /* 放大曲線與基準點 (Magnified curve and reference point) */
    transform-origin: bottom center !important;
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.3s ease,
                background-color 0.2s ease !important;
}

div.stButton > button, div.stButton > button p, div.stButton > button span {
    color: #1c1c1e !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* 按鈕 Hover 懸浮放大效果 (Button Hover Zoom Effect) */
div.stButton > button:hover {
    background-color: #f2f2f7 !important;
    transform: translateY(-5px) scale(1.08) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12) !important;
    z-index: 10 !important;
}

/* 按鈕按下時的回彈感 (The tactile feedback when the button is pressed) */
div.stButton > button:active {
    transform: translateY(-1px) scale(0.96) !important;
    transition: transform 0.1s ease !important;
}

/* 主要按鈕 (Start) 特殊 Dock 光澤與放大 (Main button (Start) with special Dock finish and magnification) */
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

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 10px 24px rgba(0, 122, 255, 0.45) !important;
}
</style>
""")

st.markdown(custom_css, unsafe_allow_html=True)

DICE_ICONS = {1: '⚀', 2: '⚁', 3: '⚂', 4: '⚃', 5: '⚄', 6: '⚅'}
X_VALUES = np.arange(2, 13)
WAYS = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
THEORETICAL_PMF = WAYS / 36
THEORY_P_X7 = 6 / 36


def render_dice_html(rolls):
    """顯示兩顆骰子的目前結果。"""
    die1, die2 = int(rolls[0]), int(rolls[1])
    total = die1 + die2

    return f"""
    <div class="dice-wrapper">
        <div class="dice-card">
            <span class="dice-icon">{DICE_ICONS[die1]}</span>
            <span class="dice-label">第一顆骰子<br>Die 1 = {die1}</span>
        </div>
        <div style="font-size:34px;font-weight:700;color:#8e8e93;">+</div>
        <div class="dice-card">
            <span class="dice-icon">{DICE_ICONS[die2]}</span>
            <span class="dice-label">第二顆骰子<br>Die 2 = {die2}</span>
        </div>
        <div style="font-size:34px;font-weight:700;color:#8e8e93;">=</div>
        <div class="dice-card matched">
            <span style="font-size:18px;font-weight:800;">X</span>
            <span class="dice-icon" style="font-size:40px;">{total}</span>
            <span class="dice-label">點數和<br>Sum = {total}</span>
        </div>
    </div>
    """


def render_kpi_html(title_cn, title_en, val, color="#1c1c1e"):
    return textwrap.dedent(f"""
    <div class="ios-kpi-card">
        <div class="ios-kpi-title">{title_cn}<br><span style="font-size:10px;">{title_en}</span></div>
        <div class="ios-kpi-value" style="color: {color};">{val}</div>
    </div>
    """)


@st.cache_data
def run_simulation(total_n, seed):
    """
    模擬擲兩顆公平六面骰 total_n 次。
    X = 第一顆骰子 + 第二顆骰子。
    """
    rng = np.random.default_rng(seed)

    die1 = rng.integers(1, 7, size=total_n)
    die2 = rng.integers(1, 7, size=total_n)
    X = die1 + die2

    # 每次累積到目前為止，各 X 值的出現次數
    cumulative_counts = np.zeros((total_n, 11), dtype=int)

    running_counts = np.zeros(11, dtype=int)
    for i, x in enumerate(X):
        running_counts[x - 2] += 1
        cumulative_counts[i] = running_counts

    cumulative_relative = cumulative_counts / np.arange(1, total_n + 1)[:, None]

    # 特別保留 X=7 的收斂軌跡
    x7_index = 7 - 2
    cum_x7 = cumulative_counts[:, x7_index]
    cum_p_x7 = cumulative_relative[:, x7_index]

    return die1, die2, X, cumulative_counts, cumulative_relative, cum_x7, cum_p_x7


def build_clean_plotly_chart(df_data, total_n_setting):
    """觀察 P(X=7) 的收斂軌跡。"""
    df_reset = df_data.reset_index().rename(columns={'index': 'n'})

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_reset['n'],
        y=df_reset['理論 P(X=7)'],
        mode='lines',
        name='理論 P(X=7) / Theory',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        hovertemplate='理論 P(X=7): %{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df_reset['n'],
        y=df_reset['模擬 P(X=7)'],
        mode='lines+markers',
        name='模擬 P(X=7) / Simulation',
        line=dict(color='#007AFF', width=2.2),
        marker=dict(size=4, color='#007AFF', opacity=0.85),
        hovertemplate='模擬次數 n: %{x}<br>模擬 P(X=7): %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        hovermode='x unified',
        xaxis=dict(
            title='模擬次數 (n) / Total Simulations (n)',
            range=[1, max(total_n_setting, 10)],
            showgrid=True,
            gridcolor='#f2f2f7',
            zeroline=False
        ),
        yaxis=dict(
            title='P(X=7) / Relative Frequency',
            range=[0, 0.35],
            showgrid=True,
            gridcolor='#f2f2f7',
            zeroline=False,
            tickformat='.3f'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        uirevision='constant'
    )
    return fig


def build_distribution_chart(relative_frequency):
    """作業第 (c) 題：理論 PMF 與模擬相對次數同圖。"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=X_VALUES,
        y=THEORETICAL_PMF,
        name='理論機率 / Theoretical PMF',
        opacity=0.55,
        marker_color='#FF3B30',
        hovertemplate='X=%{x}<br>理論機率=%{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=X_VALUES,
        y=relative_frequency,
        mode='lines+markers',
        name='模擬相對次數 / Simulation',
        line=dict(color='#007AFF', width=3),
        marker=dict(size=8, color='#007AFF'),
        hovertemplate='X=%{x}<br>模擬相對次數=%{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        hovermode='x unified',
        title='X 的機率分布與模擬結果',
        xaxis=dict(
            title='X = 兩顆骰子的點數和',
            tickmode='linear',
            dtick=1,
            showgrid=True,
            gridcolor='#f2f2f7'
        ),
        yaxis=dict(
            title='機率 / 相對次數',
            range=[0, 0.22],
            tickformat='.3f',
            showgrid=True,
            gridcolor='#f2f2f7'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


def calculate_theoretical_probability():
    """X=7 的理論機率。兩顆骰子共有 36 種等可能結果，其中 6 種和為 7。"""
    return 6 / 36


def render_pmf_table():
    pmf_df = pd.DataFrame({
        'X': X_VALUES,
        '組合數 / Ways': WAYS,
        '理論機率 / P(X=x)': THEORETICAL_PMF
    })
    return pmf_df


# ============================================================
# Session State
# ============================================================

if "anim_status" not in st.session_state:
    st.session_state.anim_status = "idle"

if "current_step_idx" not in st.session_state:
    st.session_state.current_step_idx = 0

if "total_n" not in st.session_state:
    st.session_state.total_n = 1000


def render_description():
    return textwrap.dedent(f"""
<div class="card">
    <div style="font-size: 22px; font-weight: 800; margin-bottom: 8px; color: #1c1c1e;">
        🎲 兩顆六面公平骰子機率分布模擬
    </div>

    <div style="font-size: 14px; line-height: 1.7; color: #3a3a3c;">
        📌 <b>實驗規則：</b>
        每次同時擲兩顆公平六面骰，令
        <b>X = 第一顆骰子 + 第二顆骰子</b>。
        因此 X 的可能值為 <b>2, 3, 4, ..., 12</b>。<br>

        <i>
        Roll two fair six-sided dice. Let X be the sum of the two dice.
        Therefore X can take values from 2 to 12.
        </i><br>

        🎯 <b>X = 7 的理論機率：</b>
        P(X=7) = 6/36 = {THEORY_P_X7:.4f}<br>

        📊 <b>本程式依照作業要求：</b>
        (a) 顯示 X 的 p.m.f.；
        (b) 模擬擲兩顆骰子 1,000 次並比較相對次數；
        (c) 將理論機率分布與模擬結果繪製在同一張圖。
    </div>
</div>
""")


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ 控制面板 (Control Panel)")

    def sync_from_slider():
        st.session_state.total_n = st.session_state.slider_n
        st.session_state.anim_status = "idle"
        st.session_state.current_step_idx = 0

    def sync_from_input():
        st.session_state.total_n = st.session_state.input_n
        st.session_state.anim_status = "idle"
        st.session_state.current_step_idx = 0

    st.markdown("**骰子設定 (Dice Setting)**")
    st.info("本作業固定使用 **兩顆六面公平骰子**。")

    st.markdown("**模擬總次數 (N) / Total Simulations (N)**")
    col_s1, col_s2 = st.columns([3, 2])

    with col_s1:
        st.slider(
            "拉動次數 (Slider)",
            100,
            10000,
            st.session_state.total_n,
            step=100,
            key="slider_n",
            on_change=sync_from_slider,
            label_visibility="collapsed"
        )

    with col_s2:
        st.number_input(
            "輸入次數 (Number Input)",
            100,
            10000,
            st.session_state.total_n,
            step=100,
            key="input_n",
            on_change=sync_from_input,
            label_visibility="collapsed"
        )

    total_n = st.session_state.total_n

    fps = st.slider("動畫速率 (FPS) / Speed", 2, 25, 8)
    seed = st.number_input(
        "隨機種子 (Seed) / Random Seed",
        0,
        9999,
        42
    )

    st.divider()

    col_b1, col_b2, col_b3 = st.columns(3)

    start_click = col_b1.button(
        "🚀 開始 Start",
        type="primary",
        use_container_width=True
    )

    pause_label = (
        "▶️ 繼續 Resume"
        if st.session_state.anim_status == "paused"
        else "⏸️ 暫停 Pause"
    )

    pause_click = col_b2.button(
        pause_label,
        use_container_width=True
    )

    quick_click = col_b3.button(
        "⚡ 結算 Finish",
        use_container_width=True
    )

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


# ============================================================
# Main page
# ============================================================

st.markdown(render_description(), unsafe_allow_html=True)

st.header("📌 (a) X 的機率質量函數（p.m.f.）")

pmf_df = render_pmf_table()

st.dataframe(
    pmf_df.style.format({
        '理論機率 / P(X=x)': '{:.4f}'
    }),
    use_container_width=True,
    hide_index=True
)

st.latex(r"""
P(X=x)=
\begin{cases}
\dfrac{x-1}{36}, & 2\le x\le 7 \\[4pt]
\dfrac{13-x}{36}, & 8\le x\le 12 \\[4pt]
0, & \text{otherwise}
\end{cases}
""")


# ============================================================
# Simulation data
# ============================================================

(
    die1,
    die2,
    X,
    cumulative_counts,
    cumulative_relative,
    cum_x7,
    cum_p_x7
) = run_simulation(total_n, int(seed))

rolls = np.column_stack([die1, die2])

num_frames = min(total_n, 60)
frame_indices = np.unique(
    np.linspace(1, total_n, num=num_frames, dtype=int)
)

final_counts = cumulative_counts[-1]
final_relative = cumulative_relative[-1]

p_theoretical = calculate_theoretical_probability()


# ============================================================
# Dashboard
# ============================================================

st.subheader("📈 數據儀表板 (Data Dashboard)")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

spot_kpi1 = kpi1.empty()
spot_kpi2 = kpi2.empty()
spot_kpi3 = kpi3.empty()
spot_kpi4 = kpi4.empty()

with st.container():
    st.markdown(
        "<div style='margin-top: 10px;'><b>🎲 當前丟擲結果 (Current Roll Results)</b></div>",
        unsafe_allow_html=True
    )
    dice_spot = st.empty()

with st.container():
    st.markdown(
        "<div style='margin-top: 15px;'><b>📊 P(X=7) 相對頻率收斂軌跡 (Convergence Trajectory)</b></div>",
        unsafe_allow_html=True
    )
    chart_spot = st.empty()

df_chart = pd.DataFrame({
    '模擬 P(X=7)': cum_p_x7,
    '理論 P(X=7)': p_theoretical
}, index=np.arange(1, total_n + 1))

plotly_config = {
    'scrollZoom': True,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d']
}


def render_frame_ui(frame_n):
    idx = frame_n - 1

    current_x = int(X[idx])
    current_die1 = int(die1[idx])
    current_die2 = int(die2[idx])

    spot_kpi1.markdown(
        render_kpi_html(
            "模擬次數",
            "Simulations",
            f"{frame_n}"
        ),
        unsafe_allow_html=True
    )

    spot_kpi2.markdown(
        render_kpi_html(
            "目前 X",
            "Current X",
            f"{current_x}",
            "#007AFF"
        ),
        unsafe_allow_html=True
    )

    spot_kpi3.markdown(
        render_kpi_html(
            "估算 P(X=7)",
            "Estimated P(X=7)",
            f"{cum_p_x7[idx]:.4f}",
            "#007AFF"
        ),
        unsafe_allow_html=True
    )

    spot_kpi4.markdown(
        render_kpi_html(
            "理論 P(X=7)",
            "Theory P(X=7)",
            f"{p_theoretical:.4f}",
            "#FF3B30"
        ),
        unsafe_allow_html=True
    )

    dice_spot.markdown(
        render_dice_html(rolls[idx]),
        unsafe_allow_html=True
    )

    chart_spot.plotly_chart(
        build_clean_plotly_chart(
            df_chart.iloc[:frame_n],
            total_n
        ),
        use_container_width=True,
        config=plotly_config
    )


if st.session_state.anim_status == "idle":
    render_frame_ui(1)

    st.info(
        "👈 請點擊左側面板的 **「🚀 開始 / Start」** 播放動畫，"
        "或 **「⚡ 結算 / Finish」** 直接觀看結果！\n\n"
        "*Please click **'Start'** to run the animation, "
        "or **'Finish'** to see the final results!*"
    )

elif st.session_state.anim_status == "finished":
    render_frame_ui(total_n)

    final_error_x7 = abs(cum_p_x7[-1] - p_theoretical)

    st.success(
        f"🎉 模擬完成！最終估算 P(X=7) = {cum_p_x7[-1]:.4f}，"
        f"與理論值誤差僅 {final_error_x7:.4f}\n\n"
        f"*Simulation Finished! Final Estimated P(X=7) = "
        f"{cum_p_x7[-1]:.4f}, abs error = {final_error_x7:.4f}*"
    )

elif st.session_state.anim_status == "paused":
    current_n = int(frame_indices[st.session_state.current_step_idx])

    render_frame_ui(current_n)

    st.warning(
        f"⏸️ 動畫已暫停於第 {current_n} 次模擬，"
        f"點擊左側 **「▶️ 繼續 / Resume」** 可繼續播放。\n\n"
        f"*Paused at simulation #{current_n}. Click **'Resume'** to continue.*"
    )

elif st.session_state.anim_status == "running":
    progress_bar = st.progress(0)

    start_idx = st.session_state.current_step_idx

    for idx in range(start_idx, len(frame_indices)):
        st.session_state.current_step_idx = idx
        current_n = int(frame_indices[idx])

        progress_bar.progress(
            int((idx + 1) / len(frame_indices) * 100)
        )

        render_frame_ui(current_n)

        time.sleep(1 / fps)

    progress_bar.empty()

    st.session_state.anim_status = "finished"
    st.rerun()


# ============================================================
# Final distribution required by assignment
# ============================================================

if st.session_state.anim_status in ["paused", "finished"]:

    st.header("📊 (b) 模擬結果與理論機率比較")

    result_df = pd.DataFrame({
        'X': X_VALUES,
        '出現次數 / Frequency': final_counts,
        '相對次數 / Relative Frequency': final_relative,
        '理論機率 / Theoretical P(X=x)': THEORETICAL_PMF,
        '誤差 / Error': final_relative - THEORETICAL_PMF
    })

    st.dataframe(
        result_df.style.format({
            '相對次數 / Relative Frequency': '{:.4f}',
            '理論機率 / Theoretical P(X=x)': '{:.4f}',
            '誤差 / Error': '{:+.4f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.header("📈 (c) X 的機率分布圖與模擬結果")

    distribution_fig = build_distribution_chart(final_relative)

    st.plotly_chart(
        distribution_fig,
        use_container_width=True,
        config=plotly_config
    )

    st.subheader("📋 指定模擬次數統計結果 (Summary Table)")

    targets = [
        n for n in [50, 100, 250, 500, 750, 1000]
        if n <= total_n
    ]

    table_rows = []

    for n in targets:
        row = {
            '模擬次數 / Simulations (n)': n
        }

        for x in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            row[f'X={x} 相對次數'] = (
                f"{cumulative_relative[n - 1, x - 2]:.4f}"
            )

        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🔎 前 20 次模擬結果")

    show_n = min(20, total_n)

    first_20_df = pd.DataFrame({
        '第幾次 / Trial': np.arange(1, show_n + 1),
        '第一顆骰子 / Die 1': die1[:show_n],
        '第二顆骰子 / Die 2': die2[:show_n],
        'X = 點數和 / Sum': X[:show_n]
    })

    st.dataframe(
        first_20_df,
        use_container_width=True,
        hide_index=True
    )

    max_error = np.max(
        np.abs(final_relative - THEORETICAL_PMF)
    )

    mean_error = np.mean(
        np.abs(final_relative - THEORETICAL_PMF)
    )

    st.subheader("📌 理論值與模擬值誤差")

    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric(
            "最大絕對誤差",
            f"{max_error:.4f}"
        )

    with e2:
        st.metric(
            "平均絕對誤差",
            f"{mean_error:.4f}"
        )

    with e3:
        st.metric(
            "P(X=7) 模擬值",
            f"{final_relative[5]:.4f}"
        )

    st.markdown("""
    <div class="card">
        <div style="font-size:20px;font-weight:800;margin-bottom:8px;">
            📝 實驗結論
        </div>
        <div style="font-size:14px;line-height:1.8;color:#3a3a3c;">
            兩顆公平六面骰共有 36 種等可能結果。
            X 為兩顆骰子的點數和，因此 X 的可能值為 2～12。
            X=7 時有最多的 6 種組合，所以理論機率最高，
            為 6/36 = 1/6 ≈ 0.1667。
            <br><br>
            經由電腦進行隨機模擬後，各 X 值的相對次數會接近理論機率。
            模擬次數越多，通常越能看出模擬結果逐漸接近理論機率分布。
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("✅ 作業要求完成確認")

    checklist_df = pd.DataFrame({
        "作業要求": [
            "(a) 寫出 X 的 p.m.f.",
            "(b) 電腦模擬擲兩顆骰子 1,000 次",
            "(b) 列出結果",
            "(b) 比較相對次數與對應機率",
            "(c) 繪製 X 的機率分布圖",
            "(c) 模擬結果與理論結果同圖"
        ],
        "完成": ["✅", "✅", "✅", "✅", "✅", "✅"]
    })

    st.dataframe(
        checklist_df,
        use_container_width=True,
        hide_index=True
    )
