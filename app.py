import time
import base64
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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


/* iOS-style dice hover + pop animation */
.dice-hover{display:inline-flex;flex-direction:column;align-items:center;justify-content:center;transform-origin:center bottom;transition:transform 180ms cubic-bezier(.22,1,.36,1),filter 180ms ease;cursor:default;will-change:transform}
.dice-hover:hover{transform:translateY(-7px) scale(1.20);filter:drop-shadow(0 14px 18px rgba(0,0,0,.16))}
.dice-animation{display:block;transform-origin:center center;will-change:transform,opacity}
@keyframes dicePopIOS{0%{transform:scale(.72) translateY(10px);opacity:.35}45%{transform:scale(1.10) translateY(-3px);opacity:1}72%{transform:scale(.96) translateY(1px)}100%{transform:scale(1) translateY(0);opacity:1}}
.dice-animation.pop{animation:dicePopIOS 520ms cubic-bezier(.22,1,.36,1) both}
.pip-die{width:58px;height:58px;border-radius:15px;background:linear-gradient(145deg,#fff 0%,#f2f2f7 100%);border:1px solid rgba(0,0,0,.08);box-shadow:inset 0 1px 0 rgba(255,255,255,.95),0 7px 18px rgba(0,0,0,.10);display:grid;align-items:center;justify-items:center;overflow:hidden}
.pip-grid{width:78%;height:78%;display:grid;align-items:center;justify-items:center}
.pip{width:11px;height:11px;border-radius:50%;background:#1c1c1e;box-shadow:inset 0 1px 1px rgba(255,255,255,.25)}
.pip-grid.dense .pip{width:7px;height:7px}.pip-grid.ultra-dense .pip{width:5px;height:5px}
.dice-label-ios{margin-top:6px;font-size:10px;font-weight:650;color:#8e8e93;line-height:1.1;text-align:center;white-space:nowrap}
.dice-plus,.dice-equals{font-size:24px;font-weight:700;color:#8e8e93;display:flex;align-items:center;justify-content:center;padding:0 2px}
.sum-card-ios{min-width:88px;height:92px;padding:8px;border-radius:18px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:linear-gradient(135deg,#34c759 0%,#28a745 100%);box-shadow:0 10px 22px rgba(52,199,89,.35);border:none;transform-origin:center center}
.sum-title-ios{font-size:11px;font-weight:800;color:#fff;margin-bottom:2px}.sum-value-ios{font-size:26px;font-weight:800;line-height:1;color:#fff}.sum-sub-ios{font-size:9px;font-weight:650;color:rgba(255,255,255,.88);margin-top:3px}
@media (prefers-reduced-motion:reduce){.dice-hover{transition:none}.dice-animation.pop{animation:none}}

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

DICE_ICONS = {}  # 骰子面永遠只用圓點，不顯示數字

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

def render_sound_html(sound_type="finish", volume=0.35):
    """使用 Streamlit Components 播放完成提示音。"""
    volume = max(0.0, min(1.0, float(volume)))
    if sound_type != "finish" or volume <= 0:
        return

    html = f"""
    <html>
    <body style="margin:0;background:transparent;overflow:hidden;">
    <script>
    (() => {{
        try {{
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            const volume = {volume};
            const notes = [523.25, 659.25, 783.99];
            const now = ctx.currentTime;
            notes.forEach((freq, i) => {{
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                const start = now + i * 0.13;
                osc.type = "sine";
                osc.frequency.setValueAtTime(freq, start);
                gain.gain.setValueAtTime(0.0001, start);
                gain.gain.exponentialRampToValueAtTime(Math.max(volume, 0.001), start + 0.02);
                gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.22);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(start);
                osc.stop(start + 0.24);
            }});
            setTimeout(() => ctx.close(), 900);
        }} catch (e) {{
            console.log("Audio unavailable", e);
        }}
    }})();
    </script>
    </body>
    </html>
    """
    components.html(html, height=1, scrolling=False)

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


def _pip_layout(value):
    value=int(value)
    standard={
        1:(3,3,[(2,2)]),2:(3,3,[(1,1),(3,3)]),3:(3,3,[(1,1),(2,2),(3,3)]),
        4:(3,3,[(1,1),(1,3),(3,1),(3,3)]),
        5:(3,3,[(1,1),(1,3),(2,2),(3,1),(3,3)]),
        6:(3,3,[(1,1),(2,1),(3,1),(1,3),(2,3),(3,3)])}
    if value in standard:return standard[value]
    cols=max(2,int(np.ceil(np.sqrt(value))))
    rows=int(np.ceil(value/cols))
    positions=[(i//cols+1,i%cols+1) for i in range(value)]
    return rows,cols,positions

def _pip_die_html(value):
    rows,cols,positions=_pip_layout(value)
    density=' ultra-dense' if value>=49 else (' dense' if value>=16 else '')
    pos=set(positions)
    cells=[]
    for r in range(1,rows+1):
        for c in range(1,cols+1):
            cells.append('<span class="pip"></span>' if (r,c) in pos else '<span></span>')
    return f'''<div class="pip-die"><div class="pip-grid{density}" style="grid-template-columns:repeat({cols},1fr);grid-template-rows:repeat({rows},1fr);">{''.join(cells)}</div></div>'''

def render_dice_html(single_roll,animation_key="1"):
    total=int(np.sum(single_roll)); items_html=""
    for idx,raw_val in enumerate(single_roll):
        val=int(raw_val)
        items_html+=f'''<div class="dice-hover" title="第 {idx+1} 顆骰子"><div class="dice-animation pop" style="animation-delay:{idx*45}ms;animation-name:dicePopIOS;">{_pip_die_html(val)}</div><div class="dice-label-ios">第 {idx+1} 顆骰子</div></div>'''
        if idx<len(single_roll)-1:items_html+='<div class="dice-plus">+</div>'
    items_html+=f'''<div class="dice-equals">=</div><div class="dice-hover"><div class="dice-animation pop" style="animation-delay:{len(single_roll)*45}ms;animation-name:dicePopIOS;"><div class="sum-card-ios"><div class="sum-title-ios">點數和</div><div class="sum-value-ios">{total}</div><div class="sum-sub-ios">X = {total}</div></div></div></div>'''
    return f'''<div class="dice-wrapper" data-animation-key="{animation_key}">{items_html}</div>'''

def render_kpi_html(title_cn, title_en, val, color="#1c1c1e"):
    return textwrap.dedent(f"""
    <div class="ios-kpi-card">
        <div class="ios-kpi-title">{title_cn}<br><span style="font-size:10px;">{title_en}</span></div>
        <div class="ios-kpi-value" style="color: {color};">{val}</div>
    </div>
    """)

def build_clean_plotly_chart(df_data, total_n_setting, target_x_val):
    df_reset = df_data.reset_index().rename(columns={'index': 'n'})
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_reset['n'],
        y=df_reset[f'理論 P(X={target_x_val})'],
        mode='lines',
        name=f'理論 P(X={target_x_val})',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        hovertemplate=f'理論 P(X={target_x_val}): %{{y:.4f}}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df_reset['n'],
        y=df_reset[f'模擬 P(X={target_x_val})'],
        mode='lines+markers',
        name=f'模擬 P(X={target_x_val})',
        line=dict(color='#007AFF', width=2.2),
        marker=dict(size=4, color='#007AFF', opacity=0.85),
        hovertemplate=f'模擬次數 n: %{{x}}<br>模擬 P(X={target_x_val}): %{{y:.4f}}<extra></extra>'
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
            title=f'P(X={target_x_val}) / Relative Frequency',
            range=[0, min(1.0, max(0.2, theory_p_target * 2.2))],
            showgrid=True,
            gridcolor='#f2f2f7',
            zeroline=False,
            tickformat='.3f'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision='constant'
    )
    return fig

def build_distribution_chart(relative_frequency):
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
        height=450,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        hovermode='x unified',
        title=f'X 的機率分布圖與模擬結果 ({num_dice}顆 {num_sides}面骰)',
        xaxis=dict(title='X = 點數和', tickmode='linear', dtick=1 if len(X_VALUES)<30 else None, showgrid=True, gridcolor='#f2f2f7'),
        yaxis=dict(title='機率 / 相對次數', tickformat='.3f', showgrid=True, gridcolor='#f2f2f7'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# 音效設定
with st.sidebar.expander("🔊 音效設定", expanded=True):
    sound_enabled = st.checkbox("啟用骰子音效", value=True, key="sound_enabled")
    sound_volume = st.slider("音量", 0.0, 1.0, 0.35, 0.05, key="sound_volume")

# 7. 主頁面內容
st.markdown(textwrap.dedent(f"""
<div class="card">
    <div style="font-size: 22px; font-weight: 800; margin-bottom: 8px; color: #1c1c1e;">
        🎲 自訂骰子機率分布模擬實驗
    </div>
    <div style="font-size: 14px; line-height: 1.7; color: #3a3a3c;">
        📌 <b>當前實驗配置：</b> 同時投擲 <b>{num_dice}</b> 顆 <b>{num_sides}</b> 面公平骰子。<br>
        🎯 <b>點數和 X 範圍：</b> {X_VALUES[0]} ～ {X_VALUES[-1]}（共 {TOTAL_OUTCOMES:,} 種可能組合）。<br>
        🔥 <b>最常見點數和理論值：</b> P(X={target_x}) = {ways_target if 'ways_target' in locals() else WAYS[target_x_idx]}/{TOTAL_OUTCOMES} ≈ {theory_p_target:.4f}
    </div>
</div>
"""), unsafe_allow_html=True)

st.header("📌 (a) X 的機率質量函數（p.m.f.）")

pmf_df = pd.DataFrame({
    'X (點數和)': X_VALUES,
    '組合數 / Ways': WAYS,
    '理論機率 / P(X=x)': THEORETICAL_PMF
})

st.dataframe(
    pmf_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "理論機率 / P(X=x)": st.column_config.NumberColumn(format="%.4f")
    }
)

# 執行模擬數據計算
rolls, X, cumulative_counts, cumulative_relative, cum_target, cum_p_target = run_simulation(total_n, num_dice, num_sides, int(seed))

num_frames = min(total_n, 60)
frame_indices = np.unique(np.linspace(1, total_n, num=num_frames, dtype=int))

final_counts = cumulative_counts[-1]
final_relative = cumulative_relative[-1]

st.subheader("📈 數據儀表板 (Data Dashboard)")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
spot_kpi1 = kpi1.empty()
spot_kpi2 = kpi2.empty()
spot_kpi3 = kpi3.empty()
spot_kpi4 = kpi4.empty()

with st.container():
    st.markdown("<div style='margin-top: 10px;'><b>🎲 當前丟擲結果 (Current Roll Results)</b></div>", unsafe_allow_html=True)
    dice_spot = st.empty()

with st.container():
    st.markdown(f"<div style='margin-top: 15px;'><b>📊 P(X={target_x}) 相對頻率收斂軌跡 (Convergence Trajectory)</b></div>", unsafe_allow_html=True)
    chart_spot = st.empty()

df_chart = pd.DataFrame({
    f'模擬 P(X={target_x})': cum_p_target,
    f'理論 P(X={target_x})': theory_p_target
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

    spot_kpi1.markdown(render_kpi_html("模擬次數", "Simulations", f"{frame_n}"), unsafe_allow_html=True)
    spot_kpi2.markdown(render_kpi_html("目前 X", "Current X", f"{current_x}", "#007AFF"), unsafe_allow_html=True)
    spot_kpi3.markdown(render_kpi_html(f"估算 P(X={target_x})", f"Estimated P(X={target_x})", f"{cum_p_target[idx]:.4f}", "#007AFF"), unsafe_allow_html=True)
    spot_kpi4.markdown(render_kpi_html(f"理論 P(X={target_x})", f"Theory P(X={target_x})", f"{theory_p_target:.4f}", "#FF3B30"), unsafe_allow_html=True)

    dice_html = render_dice_html(rolls[idx], animation_key=f"frame-{frame_n}")
    # st.empty() 回傳的 DeltaGenerator 不一定提供 .html()；
    # 使用 markdown + unsafe_allow_html 可相容較多 Streamlit 版本，
    # 同時保留 CSS 的 hover 放大與動畫效果。
    dice_spot.markdown(dice_html, unsafe_allow_html=True)
    chart_spot.plotly_chart(
        build_clean_plotly_chart(df_chart.iloc[:frame_n], total_n, target_x),
        use_container_width=True,
        config=plotly_config
    )

# 8. 動畫與狀態控制
if "last_sound_frame" not in st.session_state:
    st.session_state.last_sound_frame = None

if st.session_state.anim_status == "idle":
    render_frame_ui(1)
    st.info("👈 請點擊左側面板的 **「🚀 開始」** 播放動畫，或 **「⚡ 結算」** 直接觀看結果！")

elif st.session_state.anim_status == "finished":
    render_frame_ui(total_n)
    if sound_enabled:
        render_sound_html("finish", sound_volume)
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
