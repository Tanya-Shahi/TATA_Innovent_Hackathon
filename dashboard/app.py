import streamlit as st
import sys
import os
from importlib import import_module
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

# Path setup
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"

if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

os.chdir(project_root)

FleetDigitalTwin = import_module("digital_twin").FleetDigitalTwin
CMAPSSSimulator = import_module("simulator").CMAPSSSimulator

# Page config
st.set_page_config(
    page_title="Edge AI Digital Twin - Aerospace MRO",
    page_icon="plane",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    border: 1px solid #2d3250;
}
.status-critical { color: #ff4b4b; font-weight: bold; }
.status-warning  { color: #ffa500; font-weight: bold; }
.status-caution  { color: #ffd700; font-weight: bold; }
.status-healthy  { color: #00c853; font-weight: bold; }
.header-title {
    font-size: 2rem;
    font-weight: bold;
    color: #ffffff;
}
.header-sub {
    color: #a0aec0;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'fleet' not in st.session_state:
    st.session_state.fleet = None
if 'simulator' not in st.session_state:
    st.session_state.simulator = None
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = {}

@st.cache_resource
def load_fleet():
    model_path  = project_root / "models" / "lstm_rul.pt"
    scaler_path = project_root / "models" / "scaler.pkl"
    config_path = project_root / "models" / "model_config.json"
    data_path   = project_root / "data"   / "raw" / "train_FD001.txt"

    fleet     = FleetDigitalTwin(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        config_path=str(config_path)
    )
    simulator = CMAPSSSimulator(data_path=str(data_path))
    return fleet, simulator

# Load
fleet, simulator = load_fleet()

# Pre-simulate fleet at different life stages
@st.cache_resource
def get_preloaded_fleet():
    f, s = load_fleet()
    engines      = s.get_sample_engines(n=5)
    life_stages  = [0.2, 0.4, 0.55, 0.7, 0.85]

    for engine_id, stage in zip(engines, life_stages):
        f.add_engine(engine_id)
        all_data = s.get_engine_data(engine_id)
        cutoff   = int(len(all_data) * stage)
        for sensor_data in all_data[:cutoff]:
            f.ingest(engine_id, sensor_data)
    return f, s

fleet, simulator = get_preloaded_fleet()

# Header
st.markdown('<p class="header-title">Edge AI-Powered Digital Twin</p>', unsafe_allow_html=True)
st.markdown('<p class="header-sub">Aerospace MRO Predictive Maintenance Platform | Tata Technologies InnoVent</p>', unsafe_allow_html=True)
st.divider()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Fleet Overview", "Engine Deep Dive", "Live Simulation", "Alerts", "Model Performance"]
)
st.sidebar.divider()
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Algorithm: LSTM")
st.sidebar.markdown("- RMSE: 14.76 cycles")
st.sidebar.markdown("- MAE: 11.30 cycles")
st.sidebar.markdown("- Dataset: NASA CMAPSS FD001")
st.sidebar.markdown("- Inference: CPU (Edge)")

# ─── PAGE 1: Fleet Overview ───────────────────────────────────────────────────
if page == "Fleet Overview":
    st.subheader("Fleet Health Overview")

    summary = fleet.get_fleet_summary()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Engines", summary['total_engines'])
    c2.metric("Healthy",       summary['healthy'],   delta=None)
    c3.metric("Caution",       summary['caution'],   delta=None)
    c4.metric("Warning",       summary['warning'],   delta=None)
    c5.metric("Critical",      summary['critical'],  delta=None)

    st.divider()

    # Fleet status table
    st.subheader("Engine Status Board")
    fleet_status = fleet.get_fleet_status()

    for engine in fleet_status:
        status = engine['status']
        rul    = engine['current_rul']
        health = engine['health_score']
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 3])

        col1.write(f"**Engine {engine['engine_id']}**")

        if status == "CRITICAL":
            col2.markdown(f'<span class="status-critical">{status}</span>', unsafe_allow_html=True)
        elif status == "WARNING":
            col2.markdown(f'<span class="status-warning">{status}</span>', unsafe_allow_html=True)
        elif status == "CAUTION":
            col2.markdown(f'<span class="status-caution">{status}</span>', unsafe_allow_html=True)
        else:
            col2.markdown(f'<span class="status-healthy">{status}</span>', unsafe_allow_html=True)

        col3.write(f"RUL: **{rul}** cycles")
        col4.write(f"Health: **{health}%**")
        col5.progress(int(health))

    st.divider()

    # Fleet health pie chart
    st.subheader("Fleet Health Distribution")
    labels = ['Healthy', 'Caution', 'Warning', 'Critical']
    values = [summary['healthy'], summary['caution'],
              summary['warning'], summary['critical']]
    colors = ['#00c853', '#ffd700', '#ffa500', '#ff4b4b']

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        hole=0.4
    )])
    fig.update_layout(
        title="Fleet Health Distribution",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── PAGE 2: Engine Deep Dive ─────────────────────────────────────────────────
elif page == "Engine Deep Dive":
    st.subheader("Engine Deep Dive")

    engine_ids = list(fleet.engines.keys())
    selected   = st.selectbox("Select Engine", engine_ids,
                              format_func=lambda x: f"Engine {x}")

    twin  = fleet.engines[selected]
    state = twin.get_state()

    col1, col2, col3 = st.columns(3)
    col1.metric("Current RUL",   f"{state['current_rul']} cycles")
    col2.metric("Health Score",  f"{state['health_score']}%")
    col3.metric("Cycles Flown",  state['cycle'])

    st.divider()

    # RUL gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=state['health_score'],
        title={'text': f"Engine {selected} Health Score"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': "#185FA5"},
            'steps': [
                {'range': [0, 20],  'color': '#ff4b4b'},
                {'range': [20, 40], 'color': '#ffa500'},
                {'range': [40, 60], 'color': '#ffd700'},
                {'range': [60, 100],'color': '#00c853'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': state['health_score']
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # RUL history chart
    if twin.rul_history:
        st.subheader("RUL Degradation Over Time")
        rul_df = pd.DataFrame(twin.rul_history)
        fig_rul = px.line(
            rul_df, x='cycle', y='rul',
            title=f"Engine {selected} - RUL Over Cycles",
            labels={'cycle': 'Flight Cycle', 'rul': 'Remaining Useful Life'}
        )
        fig_rul.update_traces(line_color='#185FA5')
        fig_rul.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_rul, use_container_width=True)


# ─── PAGE 3: Live Simulation ──────────────────────────────────────────────────
elif page == "Live Simulation":
    st.subheader("Live Engine Simulation")
    st.write("Watch the digital twin update in real-time as flight cycles advance.")

    engine_ids  = list(fleet.engines.keys())
    sim_engine  = st.selectbox("Select Engine to Simulate",
                               engine_ids,
                               format_func=lambda x: f"Engine {x}")
    speed       = st.slider("Simulation Speed", 1, 10, 3,
                            help="Higher = faster")

    if st.button("Start Simulation"):
        all_data   = simulator.get_engine_data(sim_engine)
        total      = len(all_data)
        start      = int(total * 0.3)

        sim_fleet  = FleetDigitalTwin(
            model_path  = str(project_root / "models" / "lstm_rul.pt"),
            scaler_path = str(project_root / "models" / "scaler.pkl"),
            config_path = str(project_root / "models" / "model_config.json")
        )
        sim_fleet.add_engine(sim_engine)

        # Warm up
        for sd in all_data[:start]:
            sim_fleet.ingest(sim_engine, sd)

        placeholder = st.empty()
        rul_data    = []

        for i, sensor_data in enumerate(all_data[start:]):
            state = sim_fleet.ingest(sim_engine, sensor_data)

            if state['current_rul'] is not None:
                rul_data.append({
                    'cycle': state['cycle'],
                    'rul':   state['current_rul']
                })

                with placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current RUL",
                              f"{state['current_rul']} cycles")
                    m2.metric("Health Score",
                              f"{state['health_score']}%")
                    m3.metric("Status", state['status'])

                    if len(rul_data) > 1:
                        df = pd.DataFrame(rul_data)
                        fig = px.line(
                            df, x='cycle', y='rul',
                            title="Live RUL Tracking",
                            labels={'cycle': 'Cycle',
                                    'rul':   'RUL (cycles)'}
                        )
                        fig.update_traces(line_color='#185FA5')
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    if state['latest_alert']:
                        alert = state['latest_alert']
                        if alert['level'] == 'CRITICAL':
                            st.error(f"CRITICAL: {alert['message']}")
                        elif alert['level'] == 'WARNING':
                            st.warning(f"WARNING: {alert['message']}")
                        else:
                            st.info(f"CAUTION: {alert['message']}")

                time.sleep(1.0 / speed)


# ─── PAGE 4: Alerts ───────────────────────────────────────────────────────────
elif page == "Alerts":
    st.subheader("Maintenance Alerts")

    all_alerts = []
    for engine_id, twin in fleet.engines.items():
        for alert in twin.alerts:
            alert['engine_id'] = engine_id
            all_alerts.append(alert)

    if not all_alerts:
        st.success("No active alerts. All engines operating normally.")
    else:
        critical = [a for a in all_alerts if a['level'] == 'CRITICAL']
        warning  = [a for a in all_alerts if a['level'] == 'WARNING']
        caution  = [a for a in all_alerts if a['level'] == 'CAUTION']

        if critical:
            st.error(f"CRITICAL ALERTS ({len(critical)})")
            for a in critical:
                st.error(f"Engine {a['engine_id']} | Cycle {a['cycle']} | {a['message']}")

        if warning:
            st.warning(f"WARNING ALERTS ({len(warning)})")
            for a in warning:
                st.warning(f"Engine {a['engine_id']} | Cycle {a['cycle']} | {a['message']}")

        if caution:
            st.info(f"CAUTION ALERTS ({len(caution)})")
            for a in caution:
                st.info(f"Engine {a['engine_id']} | Cycle {a['cycle']} | {a['message']}")

        # Alert summary table
        st.divider()
        st.subheader("Alert Log")
        df_alerts = pd.DataFrame(all_alerts)[
            ['engine_id', 'level', 'message', 'cycle', 'timestamp']
        ]
        st.dataframe(df_alerts, use_container_width=True)


# ─── PAGE 5: Model Performance ────────────────────────────────────────────────
elif page == "Model Performance":
    st.subheader("Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE",      "14.76 cycles")
    col2.metric("MAE",       "11.30 cycles")
    col3.metric("Model",     "LSTM")
    col4.metric("Inference", "CPU Edge")

    st.divider()

    # Show saved plots
    eda_path   = project_root / "docs" / "eda_plots.png"
    model_path = project_root / "docs" / "model_results.png"

    if model_path.exists():
        st.subheader("Training Results")
        st.image(str(model_path), caption="Training Loss & Predicted vs Actual RUL")

    if eda_path.exists():
        st.subheader("Exploratory Data Analysis")
        st.image(str(eda_path), caption="NASA CMAPSS EDA Plots")

    st.divider()
    st.subheader("Why Edge AI?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Deployment Advantages**
        - Runs on existing CPU hardware
        - No cloud connectivity required
        - Sub-millisecond inference time
        - ONNX-ready for further optimization
        """)
    with col2:
        st.markdown("""
        **Business Impact**
        - Reduce unplanned downtime
        - Optimize maintenance scheduling
        - Cut spare parts inventory costs
        - Improve aircraft safety margins
        """)