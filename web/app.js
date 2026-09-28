// Pulso TransMi - dashboard estático (sin backend propio).
// Toda la data viene de los endpoints RPC específicos definidos en
// database/migrations/009_lock_down_anon_access.sql (api_station_list,
// api_station_summary, api_station_series, api_job_runs) - este archivo
// nunca hace SELECT directo sobre ninguna tabla.

const RECENT_CHECKS = 4;
const ACCURACY_THRESHOLD = 0.85;
const HEARTBEAT_STALE_AFTER_MINUTES = 90;
const ROLLING_WINDOW = 20;

const COLOR_PRIMARY = "#CC0211";
const COLOR_SECONDARY = "#4A4849";
const COLOR_GOOD = "#FFD100";
const COLOR_WARNING = "#F7A50C";
const COLOR_CRITICAL = "#B90000";

const supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);

let stations = [];
let summary = [];
let jobRuns = [];
let selectedStationId = null;
let map = null;
let markers = {};

function statusColor(accuracy) {
  if (accuracy === null || accuracy === undefined || Number.isNaN(accuracy)) return COLOR_SECONDARY;
  if (accuracy >= ACCURACY_THRESHOLD) return COLOR_GOOD;
  if (accuracy >= ACCURACY_THRESHOLD - 0.10) return COLOR_WARNING;
  return COLOR_CRITICAL;
}

function pct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "n/d";
  return (x * 100).toFixed(digits) + "%";
}

function byStationId(id) {
  return summary.find((s) => s.station_id === id);
}

function stationName(id) {
  const st = stations.find((s) => s.station_id === id);
  return st ? st.name : id;
}

async function loadAll() {
  const [stationsRes, summaryRes, jobsRes] = await Promise.all([
    supabase.rpc("api_station_list"),
    supabase.rpc("api_station_summary"),
    supabase.rpc("api_job_runs", { p_limit: 20 }),
  ]);
  for (const [name, res] of [["api_station_list", stationsRes], ["api_station_summary", summaryRes], ["api_job_runs", jobsRes]]) {
    if (res.error) throw new Error(`${name}: ${res.error.message}`);
  }
  stations = stationsRes.data || [];
  summary = summaryRes.data || [];
  jobRuns = jobsRes.data || [];
}

async function loadSeries(stationId, limit) {
  const { data, error } = await supabase.rpc("api_station_series", {
    p_station_id: stationId,
    p_limit: limit,
  });
  if (error) throw new Error(`api_station_series: ${error.message}`);
  return data || [];
}

function rollingAccuracy(series, window = ROLLING_WINDOW) {
  const out = new Array(series.length).fill(null);
  for (let i = window - 1; i < series.length; i++) {
    let absErr = 0;
    let absDem = 0;
    for (let j = i - window + 1; j <= i; j++) {
      absErr += Math.abs(series[j].demand - series[j].prediction);
      absDem += Math.abs(series[j].demand);
    }
    out[i] = Math.max(0, 1 - absErr / Math.max(absDem, 1));
  }
  return out;
}

// ---------- Tabs ----------
function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "station") {
        // Leaflet and Plotly both measure their container's size at
        // creation time - since these were first rendered while this tab
        // was display:none (hidden by default at load), they need an
        // explicit resize once the tab actually becomes visible, or they
        // render at zero/broken size.
        setTimeout(() => {
          map && map.invalidateSize();
          const seriesEl = document.getElementById("chart-series");
          const rollingEl = document.getElementById("chart-rolling");
          if (seriesEl && seriesEl.data) Plotly.Plots.resize(seriesEl);
          if (rollingEl && rollingEl.data) Plotly.Plots.resize(rollingEl);
        }, 50);
      }
    });
  });
}

// ---------- Overview tab ----------
function renderOverview() {
  const validAcc = summary.map((s) => s.accuracy_all_time).filter((a) => a !== null && a !== undefined);
  const overallAccuracy = validAcc.reduce((a, b) => a + b, 0) / (validAcc.length || 1);
  const totalObservations = summary.reduce((a, s) => a + Number(s.n_points || 0), 0);

  document.getElementById("kpi-accuracy").textContent = pct(overallAccuracy);
  document.getElementById("kpi-stations").textContent = summary.length;
  document.getElementById("kpi-observations").textContent = totalObservations.toLocaleString("es-CO");

  const lastRun = jobRuns.find((j) => j.job_name === "collector");
  const heartbeatStale = !lastRun || (Date.now() - new Date(lastRun.started_at).getTime()) > HEARTBEAT_STALE_AFTER_MINUTES * 60000;
  const hbEl = document.getElementById("kpi-heartbeat");
  hbEl.textContent = heartbeatStale ? "DESACTUALIZADO" : "OK";
  hbEl.style.color = heartbeatStale ? COLOR_CRITICAL : COLOR_GOOD === "#FFD100" ? "#8a6d00" : COLOR_GOOD;

  const latestObservedAt = summary.reduce((max, s) => {
    const t = s.last_observed_at ? new Date(s.last_observed_at).getTime() : 0;
    return Math.max(max, t);
  }, 0);
  const feedLagDays = Math.floor((Date.now() - latestObservedAt) / 86400000);
  document.getElementById("feed-caption").textContent =
    `Última observación real registrada: ${new Date(latestObservedAt).toISOString().slice(0, 16).replace("T", " ")} UTC ` +
    `(~${feedLagDays} días detrás del reloj real — el feed de la API es un reloj virtual que reproduce datos históricos, no un feed detenido).`;

  // Bar chart (Plotly)
  const sorted = [...summary].sort((a, b) => a.accuracy_all_time - b.accuracy_all_time);
  const trace = {
    type: "bar",
    orientation: "h",
    x: sorted.map((s) => s.accuracy_all_time),
    y: sorted.map((s) => stationName(s.station_id)),
    marker: { color: sorted.map((s) => statusColor(s.accuracy_all_time)) },
    text: sorted.map((s) => pct(s.accuracy_all_time)),
    textposition: "outside",
  };
  const layout = {
    xaxis: { title: "Accuracy (1 - WAPE)", tickformat: ".0%" },
    height: 450,
    margin: { l: 220, r: 40, t: 10, b: 40 },
    shapes: [{
      type: "line", x0: ACCURACY_THRESHOLD, x1: ACCURACY_THRESHOLD, y0: 0, y1: 1, yref: "paper",
      line: { color: COLOR_SECONDARY, dash: "dash" },
    }],
    template: "plotly_white",
  };
  Plotly.newPlot("chart-bar", [trace], layout, { responsive: true, displayModeBar: false });

  // Full table
  const thead = document.querySelector("#table-overview thead");
  const tbody = document.querySelector("#table-overview tbody");
  thead.innerHTML = "<tr><th>Estación</th><th>Nombre</th><th>Puntos</th><th>Accuracy histórica</th>" +
    "<th>Últimos 20</th><th>Últimos 4</th><th>Elegible drift</th><th>Última observación</th></tr>";
  tbody.innerHTML = [...summary].sort((a, b) => b.accuracy_all_time - a.accuracy_all_time).map((s) => `
    <tr>
      <td>${s.station_id}</td>
      <td>${stationName(s.station_id)}</td>
      <td>${Number(s.n_points).toLocaleString("es-CO")}</td>
      <td>${pct(s.accuracy_all_time)}</td>
      <td>${pct(s.accuracy_last_20)}</td>
      <td>${pct(s.accuracy_last_4)}</td>
      <td>${s.accuracy_last_4 < ACCURACY_THRESHOLD ? "Sí" : "No"}</td>
      <td>${s.last_observed_at ? new Date(s.last_observed_at).toISOString().slice(0, 16).replace("T", " ") : "n/d"}</td>
    </tr>`).join("");
}

// ---------- Station tab ----------
function initMap() {
  map = L.map("map").setView([4.645, -74.095], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  stations.forEach((st) => {
    const s = byStationId(st.station_id);
    const color = statusColor(s ? s.accuracy_last_4 : null);
    const marker = L.circleMarker([st.latitude, st.longitude], {
      radius: 11,
      color: "#1a1a1a",
      weight: 1,
      fillColor: color,
      fillOpacity: 0.9,
    }).addTo(map);
    marker.bindTooltip(`<b>${st.name}</b> (${st.station_id})`, { permanent: false });
    marker.bindPopup(
      `<b>${st.name}</b> (${st.station_id})<br>` +
      `Accuracy histórica: ${s ? pct(s.accuracy_all_time) : "n/d"}<br>` +
      `Accuracy últimos 4: ${s ? pct(s.accuracy_last_4) : "n/d"}`
    );
    marker.on("click", () => selectStation(st.station_id));
    markers[st.station_id] = marker;
  });
}

function populateStationSelect() {
  const select = document.getElementById("station-select");
  select.innerHTML = [...stations]
    .sort((a, b) => a.station_id.localeCompare(b.station_id))
    .map((st) => `<option value="${st.station_id}">${st.station_id} — ${st.name}</option>`)
    .join("");
  select.addEventListener("change", () => selectStation(select.value, { fromSelect: true }));
}

async function selectStation(stationId, opts = {}) {
  selectedStationId = stationId;
  document.getElementById("station-select").value = stationId;

  Object.entries(markers).forEach(([id, marker]) => {
    marker.setStyle({ weight: id === stationId ? 3 : 1, color: id === stationId ? COLOR_PRIMARY : "#1a1a1a" });
  });
  if (!opts.fromSelect && markers[stationId]) {
    map.panTo(markers[stationId].getLatLng());
  }

  const s = byStationId(stationId);
  document.getElementById("st-accuracy-all").textContent = pct(s.accuracy_all_time);
  document.getElementById("st-accuracy-20").textContent = pct(s.accuracy_last_20);
  document.getElementById("st-accuracy-4").textContent = pct(s.accuracy_last_4);
  document.getElementById("st-npoints").textContent = Number(s.n_points).toLocaleString("es-CO");

  const warningEl = document.getElementById("st-drift-warning");
  if (s.accuracy_last_4 !== null && s.accuracy_last_4 < ACCURACY_THRESHOLD) {
    warningEl.hidden = false;
    warningEl.textContent = `Esta estación está actualmente por debajo del umbral de reentrenamiento ` +
      `(${(ACCURACY_THRESHOLD * 100).toFixed(0)}%) en sus últimos ${RECENT_CHECKS} puntos.`;
  } else {
    warningEl.hidden = true;
  }

  await renderStationCharts(stationId);
}

async function renderStationCharts(stationId) {
  const limit = Number(document.getElementById("window-slider").value);
  const series = await loadSeries(stationId, limit);

  Plotly.newPlot("chart-series", [
    {
      x: series.map((r) => r.observed_at), y: series.map((r) => Number(r.demand)),
      name: "Real", type: "scatter", mode: "lines", line: { color: COLOR_PRIMARY, width: 2 },
    },
    {
      x: series.map((r) => r.observed_at), y: series.map((r) => Number(r.prediction)),
      name: "Predicción (nowcast h1)", type: "scatter", mode: "lines",
      line: { color: COLOR_SECONDARY, width: 2, dash: "dash" },
    },
  ], {
    title: "Demanda real vs. predicha", height: 400,
    margin: { l: 50, r: 20, t: 40, b: 40 },
    legend: { orientation: "h", y: 1.1 },
    template: "plotly_white",
  }, { responsive: true, displayModeBar: false });

  const rolling = rollingAccuracy(series.map((r) => ({ demand: Number(r.demand), prediction: Number(r.prediction) })));
  Plotly.newPlot("chart-rolling", [{
    x: series.map((r) => r.observed_at), y: rolling, name: `Accuracy móvil (${ROLLING_WINDOW} pts)`,
    type: "scatter", mode: "lines", line: { color: COLOR_PRIMARY, width: 2 },
  }], {
    title: "Tendencia de accuracy en el tiempo", height: 350,
    yaxis: { tickformat: ".0%" },
    margin: { l: 50, r: 20, t: 40, b: 40 },
    showlegend: false,
    template: "plotly_white",
    shapes: [{
      type: "line", x0: 0, x1: 1, xref: "paper", y0: ACCURACY_THRESHOLD, y1: ACCURACY_THRESHOLD,
      line: { color: COLOR_SECONDARY, dash: "dash" },
    }],
  }, { responsive: true, displayModeBar: false });
}

function setupWindowSlider() {
  const slider = document.getElementById("window-slider");
  const label = document.getElementById("window-value");
  slider.value = 500;
  label.textContent = slider.value;
  slider.addEventListener("change", () => {
    label.textContent = slider.value;
    if (selectedStationId) renderStationCharts(selectedStationId);
  });
  slider.addEventListener("input", () => { label.textContent = slider.value; });
}

// ---------- Health tab ----------
function renderHealth() {
  const lastRun = jobRuns.find((j) => j.job_name === "collector");
  const el = document.getElementById("health-heartbeat");
  if (lastRun) {
    const minsAgo = Math.round((Date.now() - new Date(lastRun.started_at).getTime()) / 60000);
    const stale = minsAgo > HEARTBEAT_STALE_AFTER_MINUTES;
    el.innerHTML = `<p><b>Última corrida exitosa:</b> ${new Date(lastRun.started_at).toISOString().slice(0, 16).replace("T", " ")} UTC (hace ${minsAgo} min)</p>` +
      (stale
        ? `<p class="warning">El collector no reporta éxito hace más de ${HEARTBEAT_STALE_AFTER_MINUTES} min.</p>`
        : `<p style="color:#8a6d00">El collector está reportando corridas recientes.</p>`);
  } else {
    el.innerHTML = `<p class="warning">No hay registros de heartbeat disponibles.</p>`;
  }

  const thead = document.querySelector("#table-jobs thead");
  const tbody = document.querySelector("#table-jobs tbody");
  thead.innerHTML = "<tr><th>Job</th><th>Estado</th><th>Inicio</th><th>Fin</th></tr>";
  tbody.innerHTML = jobRuns.map((j) => `
    <tr><td>${j.job_name}</td><td>${j.status}</td>
    <td>${new Date(j.started_at).toISOString().slice(0, 16).replace("T", " ")}</td>
    <td>${j.finished_at ? new Date(j.finished_at).toISOString().slice(0, 16).replace("T", " ") : "—"}</td></tr>`
  ).join("");

  const driftEligible = summary.filter((s) => s.accuracy_last_4 !== null && s.accuracy_last_4 < ACCURACY_THRESHOLD);
  const dthead = document.querySelector("#table-drift thead");
  const dtbody = document.querySelector("#table-drift tbody");
  if (driftEligible.length === 0) {
    dthead.innerHTML = "";
    dtbody.innerHTML = `<tr><td style="color:#8a6d00">Ninguna estación está actualmente por debajo del umbral de drift.</td></tr>`;
  } else {
    dthead.innerHTML = "<tr><th>Estación</th><th>Nombre</th><th>Accuracy últimos 4</th><th>Puntos</th></tr>";
    dtbody.innerHTML = driftEligible.map((s) => `
      <tr><td>${s.station_id}</td><td>${stationName(s.station_id)}</td>
      <td>${pct(s.accuracy_last_4)}</td><td>${Number(s.n_points).toLocaleString("es-CO")}</td></tr>`
    ).join("");
  }
}

// ---------- Boot ----------
async function main() {
  setupTabs();
  try {
    await loadAll();
  } catch (err) {
    document.getElementById("loading").hidden = true;
    const errEl = document.getElementById("error");
    errEl.hidden = false;
    errEl.textContent = `No se pudo conectar a Supabase: ${err.message}`;
    return;
  }

  document.getElementById("loading").hidden = true;
  document.getElementById("content").hidden = false;

  renderOverview();
  initMap();
  populateStationSelect();
  setupWindowSlider();
  renderHealth();

  const firstStation = [...stations].sort((a, b) => a.station_id.localeCompare(b.station_id))[0];
  if (firstStation) await selectStation(firstStation.station_id, { fromSelect: true });
}

main();
