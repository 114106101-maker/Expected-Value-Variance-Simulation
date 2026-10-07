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
        st.markdown(render_sound_html("finish", sound_volume), unsafe_allow_html=True)
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
