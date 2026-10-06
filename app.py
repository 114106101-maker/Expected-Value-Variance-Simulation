```python
import time
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# 1. 頁面設定
# ============================================================

st.set_page_config(
    page_title="🎲 兩顆骰子機率分布模擬",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CSS 樣式
# ============================================================

custom_css = textwrap.dedent("""
<style>

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: #ffffff !important;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "SF Pro Text",
        "Segoe UI",
        sans-serif !important;
    color: #1c1c1e !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1150px;
}

/* 卡片 */

.card {
    background: #ffffff;
    border-radius: 20px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.06);
}

/* KPI */

.kpi-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 14px 10px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.06);
}

.kpi-title {
    font-size: 12px;
    font-weight: 500;
    color: #8e8e93;
    margin-bottom: 4px;
    line-height: 1.3;
}

.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #1c1c1e;
}

/* 骰子 */

.dice-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 24px;
    margin: 20px 0;
    padding: 18px 0;
}

.dice-card {
    width: 120px;
    height: 120px;
    background: #ffffff;
    border-radius: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(0, 0, 0, 0.08);
}

.dice-icon {
    font-size: 55px;
    line-height: 1;
    margin-bottom: 8px;
}

.dice-label {
    font-size: 13px;
    font-weight: 600;
    color: #8e8e93;
    text-align: center;
}

/* 和 */

.sum-card {
    width: 150px;
    height: 120px;
    background: #f2f2f7;
    border-radius: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.06);
}

.sum-number {
    font-size: 42px;
    font-weight: 800;
    color: #1c1c1e;
}

.sum-label {
    font-size: 13px;
    font-weight: 600;
    color: #8e8e93;
}

/* 按鈕 */

div.stButton > button {
    border-radius: 14px !important;
    padding: 0.6rem 0.5rem !important;
    font-weight: 600 !important;
}

</style>
""")

st.markdown(custom_css, unsafe_allow_html=True)


# ============================================================
# 3. 基本資料
# ============================================================

DICE_ICONS = {
    1: "⚀",
    2: "⚁",
    3: "⚂",
    4: "⚃",
    5: "⚄",
    6: "⚅"
}

X_VALUES = np.arange(2, 13)

# 兩顆骰子加總後的組合數
WAYS = np.array([
    1, 2, 3, 4, 5,
    6,
    5, 4, 3, 2, 1
])

THEORETICAL_PROBABILITY = WAYS / 36


# ============================================================
# 4. 理論機率函數
# ============================================================

def theoretical_pmf(x):
    """
    兩顆公平六面骰的和 X 的理論機率。
    """
    if 2 <= x <= 7:
        return (x - 1) / 36
    elif 8 <= x <= 12:
        return (13 - x) / 36
    else:
        return 0


# ============================================================
# 5. 顯示骰子
# ============================================================

def render_dice_html(die1, die2, total):
    return f"""
    <div class="dice-wrapper">

        <div class="dice-card">
            <div class="dice-icon">{DICE_ICONS[die1]}</div>
            <div class="dice-label">
                第一顆骰子<br>
                Die 1 = {die1}
            </div>
        </div>

        <div style="
            font-size: 36px;
            font-weight: 700;
            color: #8e8e93;
        ">
            +
        </div>

        <div class="dice-card">
            <div class="dice-icon">{DICE_ICONS[die2]}</div>
            <div class="dice-label">
                第二顆骰子<br>
                Die 2 = {die2}
            </div>
        </div>

        <div style="
            font-size: 36px;
            font-weight: 700;
            color: #8e8e93;
        ">
            =
        </div>

        <div class="sum-card">
            <div class="sum-number">{total}</div>
            <div class="sum-label">
                X = Dice 1 + Dice 2
            </div>
        </div>

    </div>
    """


# ============================================================
# 6. KPI
# ============================================================

def render_kpi(title_cn, title_en, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">
            {title_cn}<br>
            <span style="font-size:10px;">
                {title_en}
            </span>
        </div>

        <div class="kpi-value">
            {value}
        </div>
    </div>
    """


# ============================================================
# 7. 模擬函數
# ============================================================

@st.cache_data
def run_simulation(total_n, seed):

    rng = np.random.default_rng(seed)

    # 每一次實驗都擲兩顆公平六面骰
    dice1 = rng.integers(1, 7, size=total_n)
    dice2 = rng.integers(1, 7, size=total_n)

    # X = 兩顆骰子的點數和
    X = dice1 + dice2

    # 統計 X = 2~12 各出現幾次
    counts = np.array([
        np.sum(X == x)
        for x in X_VALUES
    ])

    # 相對次數
    relative_frequency = counts / total_n

    return dice1, dice2, X, counts, relative_frequency


# ============================================================
# 8. 圖表
# ============================================================

def build_distribution_chart(
    values,
    theoretical_probability,
    relative_frequency
):

    fig = go.Figure()

    # 理論機率
    fig.add_trace(
        go.Bar(
            x=values,
            y=theoretical_probability,
            name="理論機率 / Theoretical Probability",
            opacity=0.65
        )
    )

    # 模擬相對次數
    fig.add_trace(
        go.Scatter(
            x=values,
            y=relative_frequency,
            mode="lines+markers",
            name="模擬相對次數 / Simulation",
            line=dict(width=3),
            marker=dict(size=8)
        )
    )

    fig.update_layout(
        height=500,
        title="X 的機率分布與模擬結果",
        xaxis_title="X = 兩顆骰子的點數和",
        yaxis_title="機率 / 相對次數",
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        yaxis=dict(
            range=[0, max(0.20, max(
                theoretical_probability.max(),
                relative_frequency.max()
            ) + 0.03)],
            tickformat=".3f"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    return fig


# ============================================================
# 9. Session State
# ============================================================

if "anim_status" not in st.session_state:
    st.session_state.anim_status = "idle"

if "current_step_idx" not in st.session_state:
    st.session_state.current_step_idx = 0

if "total_n" not in st.session_state:
    st.session_state.total_n = 1000


# ============================================================
# 10. 側邊欄
# ============================================================

with st.sidebar:

    st.header("⚙️ 模擬控制面板")

    st.markdown("### 🎲 模擬設定")

    total_n = st.number_input(
        "模擬次數 N",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=42,
        step=1
    )

    fps = st.slider(
        "動畫速度 / FPS",
        min_value=1,
        max_value=20,
        value=8
    )

    st.divider()

    st.markdown("""
    **本次作業設定**

    - 骰子數量：2 顆
    - 每顆骰子：6 面
    - 骰子：公平骰子
    - 模擬次數：1,000 次
    - X：兩顆骰子點數之和
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        start_click = st.button(
            "🚀 開始",
            use_container_width=True
        )

    with col2:
        finish_click = st.button(
            "⚡ 結算",
            use_container_width=True
        )

    if start_click:
        st.session_state.anim_status = "running"
        st.session_state.current_step_idx = 0
        st.rerun()

    if finish_click:
        st.session_state.anim_status = "finished"
        st.rerun()


# ============================================================
# 11. 題目說明
# ============================================================

st.markdown("""
<div class="card">

<h2>🎲 兩顆六面公平骰子的和</h2>

<p style="font-size:16px; line-height:1.8;">

擲兩個六面的公平骰子，令

<b>X = 第一顆骰子的結果 + 第二顆骰子的結果</b>

因此 X 的可能值為：

<b>2, 3, 4, ..., 12</b>

本程式透過電腦模擬擲兩顆骰子 1,000 次，
並比較「理論機率」與「模擬相對次數」。

</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 12. (a) PMF
# ============================================================

st.header("📌 (a) X 的機率質量函數（p.m.f.）")

st.markdown("""
兩顆公平六面骰共有：

\[
6 \\times 6 = 36
\]

種等可能的結果。
""")

pmf_df = pd.DataFrame({
    "X": X_VALUES,
    "組合數": WAYS,
    "理論機率 P(X=x)": THEORETICAL_PROBABILITY
})

st.dataframe(
    pmf_df.style.format({
        "理論機率 P(X=x)": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("### p.m.f. 數學表示")

st.latex(r"""
P(X=x)=
\begin{cases}
\dfrac{x-1}{36}, & 2\leq x\leq7 \\[6pt]
\dfrac{13-x}{36}, & 8\leq x\leq12 \\[6pt]
0, & \text{otherwise}
\end{cases}
""")


# ============================================================
# 13. 模擬資料
# ============================================================

dice1, dice2, X, counts, relative_frequency = run_simulation(
    int(total_n),
    int(seed)
)

max_error = np.max(
    np.abs(
        relative_frequency -
        THEORETICAL_PROBABILITY
    )
)

mean_error = np.mean(
    np.abs(
        relative_frequency -
        THEORETICAL_PROBABILITY
    )
)


# ============================================================
# 14. (b) 模擬
# ============================================================

st.header("🎲 (b) 電腦模擬擲兩顆骰子")

st.markdown(f"""
目前設定：

- 模擬次數：**{int(total_n):,} 次**
- 第一顆骰子：1～6
- 第二顆骰子：1～6
- 每次實驗計算：
  
  \[
  X = Dice_1 + Dice_2
  \]
""")


# ============================================================
# 15. 動畫顯示
# ============================================================

st.subheader("🎲 模擬過程")

dice_spot = st.empty()
progress_spot = st.empty()
message_spot = st.empty()

# 動畫顯示次數
frame_count = min(int(total_n), 50)

frame_indices = np.unique(
    np.linspace(
        1,
        int(total_n),
        frame_count,
        dtype=int
    )
)


def render_frame(frame_n):

    index = frame_n - 1

    dice_spot.markdown(
        render_dice_html(
            int(dice1[index]),
            int(dice2[index]),
            int(X[index])
        ),
        unsafe_allow_html=True
    )

    message_spot.info(
        f"""
        第 **{frame_n:,}** 次模擬

        第一顆骰子 = **{dice1[index]}**

        第二顆骰子 = **{dice2[index]}**

        因此：

        **X = {dice1[index]} + {dice2[index]} = {X[index]}**
        """
    )


if st.session_state.anim_status == "idle":

    render_frame(1)

    st.info(
        "👈 請按左側「🚀 開始」觀看模擬過程，"
        "或按「⚡ 結算」直接查看 1,000 次模擬結果。"
    )


elif st.session_state.anim_status == "running":

    progress_bar = progress_spot.progress(0)

    for i, frame_n in enumerate(frame_indices):

        render_frame(frame_n)

        progress_bar.progress(
            int((i + 1) / len(frame_indices) * 100)
        )

        time.sleep(1 / fps)

    progress_spot.empty()

    st.session_state.anim_status = "finished"

    st.rerun()


elif st.session_state.anim_status == "finished":

    render_frame(int(total_n))

    st.success(
        f"🎉 已完成 {int(total_n):,} 次模擬！"
    )


# ============================================================
# 16. 結果統計表
# ============================================================

st.subheader("📊 模擬統計結果")

result_df = pd.DataFrame({
    "X": X_VALUES,
    "出現次數": counts,
    "相對次數": relative_frequency,
    "理論機率": THEORETICAL_PROBABILITY,
    "誤差": (
        relative_frequency -
        THEORETICAL_PROBABILITY
    )
})

st.dataframe(
    result_df.style.format({
        "相對次數": "{:.4f}",
        "理論機率": "{:.4f}",
        "誤差": "{:+.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 17. 前 20 次結果
# ============================================================

st.subheader("🔎 前 20 次模擬結果")

show_n = min(20, int(total_n))

sample_df = pd.DataFrame({
    "第幾次": np.arange(1, show_n + 1),
    "第一顆骰子": dice1[:show_n],
    "第二顆骰子": dice2[:show_n],
    "X = 兩顆骰子之和": X[:show_n]
})

st.dataframe(
    sample_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 18. (c) 機率分布圖
# ============================================================

st.header("📈 (c) X 的機率分布圖與模擬結果")

fig = build_distribution_chart(
    X_VALUES,
    THEORETICAL_PROBABILITY,
    relative_frequency
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 19. 誤差摘要
# ============================================================

st.subheader("📌 理論值與模擬值比較")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        render_kpi(
            "模擬次數",
            "Simulations",
            f"{int(total_n):,}"
        ),
        unsafe_allow_html=True
    )

with kpi2:
    st.markdown(
        render_kpi(
            "X 的可能值",
            "Possible X",
            "2～12"
        ),
        unsafe_allow_html=True
    )

with kpi3:
    st.markdown(
        render_kpi(
            "最大絕對誤差",
            "Max Error",
            f"{max_error:.4f}"
        ),
        unsafe_allow_html=True
    )

with kpi4:
    st.markdown(
        render_kpi(
            "平均絕對誤差",
            "Mean Error",
            f"{mean_error:.4f}"
        ),
        unsafe_allow_html=True
    )


# ============================================================
# 20. 結論
# ============================================================

st.markdown("""
<div class="card">

<h3>📝 模擬結論</h3>

<p style="font-size:15px; line-height:1.8;">

兩顆公平六面骰的點數和 X，其理論機率分布呈現對稱的三角形：

<br><br>

X = 7 時的理論機率最高：

\[
P(X=7)=\frac{6}{36}=\frac{1}{6}\approx0.1667
\]

<br>

X = 2 與 X = 12 時的理論機率最低：

\[
P(X=2)=P(X=12)=\frac{1}{36}
\]

<br>

透過電腦模擬 1,000 次，可以發現各 X 值的相對次數
通常會接近理論機率，但因為隨機抽樣的關係，
模擬結果不會完全等於理論值。

</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 21. 作業對照
# ============================================================

st.subheader("✅ 本程式與作業要求對照")

check_df = pd.DataFrame({
    "作業要求": [
        "(a) 寫出 X 的 p.m.f.",
        "(b) 模擬擲兩顆骰子 1,000 次",
        "(b) 列出模擬結果",
        "(b) 比較相對次數與理論機率",
        "(c) 繪製 X 的機率分布圖",
        "(c) 模擬結果與理論結果同圖"
    ],
    "程式完成": [
        "✅",
        "✅",
        "✅",
        "✅",
        "✅",
        "✅"
    ]
})

st.dataframe(
    check_df,
    use_container_width=True,
    hide_index=True
)
```
