"""Local read-only dashboard for Pulso TransMi.

Evaluates per-station model performance and general project statistics by
querying Supabase directly. This is purely an observability tool - it never
writes to the database and has no effect on the live collector/submission
pipeline running in GitHub Actions.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import psycopg
import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# Fondo claro fijo para todas las gráficas, independiente del modo
# claro/oscuro del navegador de quien las vea (ver también .streamlit/config.toml
# para el tema general de la app).
pio.templates.default = "plotly_white"

RECENT_CHECKS = 4
ACCURACY_THRESHOLD = 0.85
HEARTBEAT_STALE_AFTER_MINUTES = 90
ROLLING_WINDOW = 20

# Paleta inspirada en TransMilenio: rojo, amarillo y ámbar extraídos del
# archivo oficial del logo (Wikimedia Commons, File:TransMilenio_logo.svg).
# No se encontró un manual de marca público con códigos hex/Pantone
# explícitos (el "Manual de Imagen de Marca y Normas Gráficas SITP" que sí
# existe especifica un azul para el uniforme de buses del SITP, distinto del
# rojo con el que se identifica visualmente a TransMilenio) - estos son los
# tonos reales tomados directamente del archivo del logo, no inventados.
COLOR_PRIMARY = "#CC0211"    # rojo TransMilenio - series principales
COLOR_SECONDARY = "#4A4849"  # gris cálido del mismo logo - líneas de referencia
COLOR_GOOD = "#FFD100"       # amarillo TransMilenio - accuracy en o sobre el umbral
COLOR_WARNING = "#F7A50C"    # ámbar - accuracy moderadamente por debajo
COLOR_CRITICAL = "#B90000"   # rojo oscuro - accuracy muy por debajo del umbral

st.set_page_config(page_title="Pulso TransMi - Dashboard", layout="wide")


@st.cache_resource
def _connection_string() -> str:
    url = os.environ.get("SUPABASE_DATABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_DATABASE_URL no está definido en .env")
    return url


@st.cache_data(ttl=60)
def load_observations() -> pd.DataFrame:
    query = """
        SELECT station_id, observed_at, demand, prediction
        FROM "Original Data"
        WHERE prediction IS NOT NULL
        UNION ALL
        SELECT station_id, observed_at, demand, prediction
        FROM "Temp"
        WHERE prediction IS NOT NULL
        ORDER BY station_id, observed_at
    """
    with psycopg.connect(_connection_string()) as conn:
        df = pd.read_sql(query, conn)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df["abs_error"] = (df["demand"] - df["prediction"]).abs()
    return df


@st.cache_data(ttl=60)
def load_stations() -> pd.DataFrame:
    query = "SELECT station_id, name, latitude, longitude FROM stations ORDER BY station_id"
    with psycopg.connect(_connection_string()) as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=60)
def load_job_runs(limit: int = 20) -> pd.DataFrame:
    query = """
        SELECT job_name, status, started_at, finished_at
        FROM ops.job_runs
        ORDER BY started_at DESC
        LIMIT %(limit)s
    """
    with psycopg.connect(_connection_string()) as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def wape_accuracy(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    denom = max(frame["demand"].abs().sum(), 1)
    wape = frame["abs_error"].sum() / denom
    return max(0.0, 1 - wape)


def station_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("observed_at")
        rows.append(
            {
                "station_id": station_id,
                "n_points": len(g),
                "last_observed_at": g["observed_at"].max(),
                "accuracy_all_time": wape_accuracy(g),
                "accuracy_last_20": wape_accuracy(g.tail(20)),
                "accuracy_last_4": wape_accuracy(g.tail(RECENT_CHECKS)),
            }
        )
    out = pd.DataFrame(rows)
    out["drift_eligible"] = out["accuracy_last_4"] < ACCURACY_THRESHOLD
    return out.sort_values("accuracy_all_time", ascending=False)


def status_color(accuracy: float | None) -> str:
    if accuracy is None:
        return COLOR_SECONDARY
    if accuracy >= ACCURACY_THRESHOLD:
        return COLOR_GOOD
    if accuracy >= ACCURACY_THRESHOLD - 0.10:
        return COLOR_WARNING
    return COLOR_CRITICAL


def rolling_accuracy(g: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    g = g.sort_values("observed_at").copy()
    denom = g["demand"].abs().rolling(window, min_periods=window).sum().clip(lower=1)
    num = g["abs_error"].rolling(window, min_periods=window).sum()
    g["rolling_accuracy"] = (1 - num / denom).clip(lower=0)
    return g


st.title("Pulso TransMi — Panel de desempeño")
st.caption(
    "Herramienta local de solo lectura sobre Supabase. No afecta el pipeline "
    "en GitHub Actions. Las métricas de accuracy aquí se calculan sobre el "
    "nowcast h1 registrado para monitoreo de drift (Original Data + Temp), "
    "no sobre las 4 predicciones reales enviadas en cada submission — ese "
    "detalle no se persiste hoy en la base de datos."
)

try:
    obs = load_observations()
    stations = load_stations()
    job_runs = load_job_runs()
except Exception as exc:  # noqa: BLE001 - surface connection errors plainly
    st.error(f"No se pudo conectar a Supabase: {exc}")
    st.stop()

if obs.empty:
    st.warning("No hay observaciones con predicción registrada todavía.")
    st.stop()

summary = station_summary(obs)
name_by_station = dict(zip(stations["station_id"], stations["name"]))
summary["name"] = summary["station_id"].map(name_by_station)

overall_accuracy = summary["accuracy_all_time"].mean()
latest_observed_at = obs["observed_at"].max()
now_utc = dt.datetime.now(dt.timezone.utc)
feed_lag_days = (now_utc - latest_observed_at).days

last_collector_run = job_runs[job_runs["job_name"] == "collector"].head(1)
if not last_collector_run.empty:
    last_run_time = last_collector_run.iloc[0]["started_at"]
    heartbeat_stale = (now_utc - last_run_time) > dt.timedelta(
        minutes=HEARTBEAT_STALE_AFTER_MINUTES
    )
else:
    last_run_time = None
    heartbeat_stale = True

tab_overview, tab_station, tab_health = st.tabs(
    ["Resumen general", "Detalle por estación", "Drift y salud del pipeline"]
)

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy promedio (todas las estaciones)", f"{overall_accuracy:.1%}")
    col2.metric("Estaciones con datos", f"{summary['station_id'].nunique()}")
    col3.metric("Observaciones con predicción", f"{len(obs):,}")
    col4.metric(
        "Collector",
        "OK" if not heartbeat_stale else "DESACTUALIZADO",
        delta=None,
    )

    st.caption(
        f"Última observación real registrada: {latest_observed_at:%Y-%m-%d %H:%M UTC} "
        f"(~{feed_lag_days} días detrás del reloj real — el feed de la API es un "
        f"reloj virtual que reproduce datos históricos, no un feed detenido)."
    )

    st.subheader("Accuracy por estación (histórico completo)")
    bar_df = summary.sort_values("accuracy_all_time", ascending=True)
    colors = [status_color(a) for a in bar_df["accuracy_all_time"]]
    fig = go.Figure(
        go.Bar(
            x=bar_df["accuracy_all_time"],
            y=bar_df["name"].fillna(bar_df["station_id"]),
            orientation="h",
            marker_color=colors,
            text=[f"{a:.1%}" for a in bar_df["accuracy_all_time"]],
            textposition="outside",
        )
    )
    fig.add_vline(
        x=ACCURACY_THRESHOLD,
        line_dash="dash",
        line_color=COLOR_SECONDARY,
        annotation_text=f"umbral de drift ({ACCURACY_THRESHOLD:.0%})",
    )
    fig.update_layout(
        xaxis_title="Accuracy (1 - WAPE)",
        xaxis_tickformat=".0%",
        yaxis_title=None,
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Tabla completa por estación")
    display_cols = [
        "station_id",
        "name",
        "n_points",
        "accuracy_all_time",
        "accuracy_last_20",
        "accuracy_last_4",
        "drift_eligible",
        "last_observed_at",
    ]
    st.dataframe(
        summary[display_cols].style.format(
            {
                "accuracy_all_time": "{:.1%}",
                "accuracy_last_20": "{:.1%}",
                "accuracy_last_4": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with tab_station:
    STATE_KEY = "selected_station_id"
    station_options = summary.sort_values("station_id").merge(
        stations[["station_id", "latitude", "longitude"]], on="station_id", how="left"
    )

    st.subheader("Mapa de estaciones (Bogotá)")
    st.caption("Haz clic en una estación del mapa para seleccionarla — el color indica su accuracy reciente (últimos 4 puntos).")

    map_colors = [status_color(a) for a in station_options["accuracy_last_4"]]
    hover_text = [
        f"<b>{row.name}</b> ({row.station_id})<br>"
        f"Accuracy histórica: {row.accuracy_all_time:.1%}<br>"
        f"Accuracy últimos 4: {row.accuracy_last_4:.1%}" if pd.notna(row.accuracy_last_4)
        else f"<b>{row.name}</b> ({row.station_id})<br>Accuracy histórica: {row.accuracy_all_time:.1%}"
        for row in station_options.itertuples()
    ]
    map_fig = go.Figure(
        go.Scattermap(
            lat=station_options["latitude"],
            lon=station_options["longitude"],
            mode="markers+text",
            marker=dict(size=22, color=map_colors),
            text=station_options["station_id"],
            textposition="top center",
            textfont=dict(color="#1a1a1a", size=11),
            hovertext=hover_text,
            hoverinfo="text",
        )
    )
    map_fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=4.645, lon=-74.095), zoom=10.3),
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
    )
    map_event = st.plotly_chart(
        map_fig, width="stretch", on_select="rerun", key="station_map_click"
    )

    ids = list(station_options["station_id"])
    default_id = st.session_state.get(STATE_KEY, ids[0])
    selection = getattr(map_event, "selection", None) or (map_event or {}).get("selection")
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if points:
        clicked_idx = points[0]["point_index"]
        default_id = ids[clicked_idx]
        st.session_state[STATE_KEY] = default_id

    labels = [
        f"{row.station_id} — {row.name}" for row in station_options.itertuples()
    ]
    default_index = ids.index(default_id) if default_id in ids else 0
    choice = st.selectbox("Estación (o elige aquí en vez del mapa)", labels, index=default_index)
    station_id = choice.split(" — ")[0]
    st.session_state[STATE_KEY] = station_id

    g = obs[obs["station_id"] == station_id].sort_values("observed_at")
    row = summary[summary["station_id"] == station_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy histórica", f"{row.accuracy_all_time:.1%}")
    col2.metric("Accuracy últimos 20", f"{row.accuracy_last_20:.1%}")
    col3.metric(
        "Accuracy últimos 4 (gate de drift)",
        f"{row.accuracy_last_4:.1%}" if pd.notna(row.accuracy_last_4) else "n/d",
    )
    col4.metric("Puntos observados", f"{row.n_points:,}")

    if row.drift_eligible:
        st.warning(
            "Esta estación está actualmente por debajo del umbral de reentrenamiento "
            f"({ACCURACY_THRESHOLD:.0%}) en sus últimos {RECENT_CHECKS} puntos."
        )

    window = st.slider("Ventana de puntos recientes a graficar", 50, min(2000, len(g)), min(500, len(g)))
    g_tail = g.tail(window)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=g_tail["observed_at"],
            y=g_tail["demand"],
            name="Real",
            line=dict(color=COLOR_PRIMARY, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g_tail["observed_at"],
            y=g_tail["prediction"],
            name="Predicción (nowcast h1)",
            line=dict(color=COLOR_SECONDARY, width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Demanda real vs. predicha",
        xaxis_title=None,
        yaxis_title="Demanda",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")

    g_roll = rolling_accuracy(g)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=g_roll["observed_at"],
            y=g_roll["rolling_accuracy"],
            name=f"Accuracy móvil ({ROLLING_WINDOW} pts)",
            line=dict(color=COLOR_PRIMARY, width=2),
        )
    )
    fig2.add_hline(
        y=ACCURACY_THRESHOLD,
        line_dash="dash",
        line_color=COLOR_SECONDARY,
        annotation_text=f"umbral de drift ({ACCURACY_THRESHOLD:.0%})",
    )
    fig2.update_layout(
        title="Tendencia de accuracy en el tiempo",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig2, width="stretch")

with tab_health:
    st.subheader("Heartbeat del collector")
    if last_run_time is not None:
        mins_ago = (now_utc - last_run_time).total_seconds() / 60
        st.metric(
            "Última corrida exitosa",
            f"{last_run_time:%Y-%m-%d %H:%M UTC}",
            delta=f"hace {mins_ago:.0f} min",
        )
        if heartbeat_stale:
            st.error(
                f"El collector no reporta éxito hace más de {HEARTBEAT_STALE_AFTER_MINUTES} min."
            )
        else:
            st.success("El collector está reportando corridas recientes.")
    else:
        st.error("No hay registros en ops.job_runs.")

    st.dataframe(job_runs, width="stretch", hide_index=True)

    st.subheader("Estaciones elegibles para reentrenamiento por drift")
    drift_df = summary[summary["drift_eligible"]][
        ["station_id", "name", "accuracy_last_4", "n_points"]
    ]
    if drift_df.empty:
        st.success("Ninguna estación está actualmente por debajo del umbral de drift.")
    else:
        st.dataframe(
            drift_df.style.format({"accuracy_last_4": "{:.1%}"}),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "El gate real de reentrenamiento en producción también exige que 'Temp' "
        "tenga datos pendientes (has_pending_data()); esta tabla solo muestra el "
        "criterio de accuracy, no el gate completo."
    )
