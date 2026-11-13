# Trend Detective - NASA Space Apps 2026
# Team Water Detectives - Deangelo Reyes, Hosur, Tamil Nadu, India
# Challenge: Be An Earth System Trend Detective!
# Run:   streamlit run app.py
# Needs: pip install streamlit streamlit-folium folium scipy requests matplotlib
# Data:  NASA POWER API (MERRA-2 reanalysis + satellite data), monthly, 1981-2025

import math, io, csv, datetime
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# ====================================================================== settings
VARIABLES = {
    "Air temperature":   ("T2M", "°C",
        "Average air temperature 2 m above the ground. Rising = the place is warming."),
    "Rainfall":          ("PRECTOTCORR", "mm/day",
        "Average rain per day in each month. Falling = drier, rising = wetter."),
    "Soil wetness (root zone)": ("GWETROOT", "0-1",
        "How wet the soil is where plant roots live. 0 = bone dry, 1 = soaked. Matters for farms and borewells."),
    "Humidity":          ("RH2M", "%",
        "Water vapour in the air as a percent of the maximum possible."),
    "Wind speed":        ("WS2M", "m/s", "Average wind speed 2 m above the ground."),
    "Sunlight on ground": ("ALLSKY_SFC_SW_DWN", "kWh/m²/day",
        "Solar energy reaching the ground per day, clouds included. Useful for solar power."),
    "Max temperature":   ("T2M_MAX", "°C", "Hottest 2 m air temperature in the month."),
    "Min temperature":   ("T2M_MIN", "°C", "Coldest 2 m air temperature in the month."),
}
ALL_CODES = ",".join(v[0] for v in VARIABLES.values())

DEMOS = {
    "My home: Hosur, India": (12.74, 77.83),
    "Arctic (80°N)": (80.0, 0.0),
    "Sahara Desert": (23.0, 10.0),
    "Amazon rain forest": (-3.0, -60.0),
    "Chennai, India": (13.08, 80.27),
    "Antarctica (coast)": (-70.0, 10.0),
}
SEASONS = {"Winter (Dec-Feb)": (12, 1, 2), "Summer (Mar-May)": (3, 4, 5),
           "Monsoon (Jun-Sep)": (6, 7, 8, 9), "Post-monsoon (Oct-Nov)": (10, 11)}

# ====================================================================== data
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_all(lat, lon, start, end):
    """One NASA POWER call for ALL variables -> {code: {(yr, mo): value}}"""
    url = ("https://power.larc.nasa.gov/api/temporal/monthly/point"
           f"?parameters={ALL_CODES}&community=AG&longitude={lon}&latitude={lat}"
           f"&start={start}&end={end}&format=JSON")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    params = r.json()["properties"]["parameter"]
    out = {}
    for code, raw in params.items():
        months = {}
        for key, v in raw.items():
            yr, mo = int(key[:4]), int(key[4:])
            if mo == 13 or v is None or v == -999:
                continue
            months[(yr, mo)] = v
        out[code] = months
    return out


def _geocode_openmeteo(name):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": 1, "language": "en", "format": "json"}, timeout=20)
    r.raise_for_status()
    res = r.json().get("results")
    if not res:
        return None
    x = res[0]
    return float(x["latitude"]), float(x["longitude"]), f'{x["name"]}, {x.get("country", "")}'


def _geocode_nominatim(name):
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": name, "format": "json", "limit": 1, "accept-language": "en"},
                     headers={"User-Agent": "TrendDetective-SpaceApps2026"}, timeout=20)
    r.raise_for_status()
    j = r.json()
    if not j:
        return None
    return float(j[0]["lat"]), float(j[0]["lon"]), j[0]["display_name"]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _geocode_cached(name):
    """Only successful answers are cached (a failure raises, so it is not stored)."""
    for fn in (_geocode_openmeteo, _geocode_nominatim):
        try:
            g = fn(name)
        except Exception:
            continue
        if g:
            return g
    raise LookupError(name)


def geocode(name):
    """City name -> (lat, lon, display name), or None. Never crashes the app."""
    try:
        return _geocode_cached(name)
    except Exception:
        return None


# ====================================================================== maths
def yearly_means(months):
    by_year = {}
    for (yr, mo), v in months.items():
        by_year.setdefault(yr, []).append(v)
    years = sorted(y for y in by_year if len(by_year[y]) == 12)
    return years, [sum(by_year[y]) / 12 for y in years]


def season_means(months, season_months):
    by_year = {}
    for (yr, mo), v in months.items():
        if mo in season_months:
            # December belongs to the following winter
            key = yr + 1 if mo == 12 else yr
            by_year.setdefault(key, []).append(v)
    years = sorted(y for y in by_year if len(by_year[y]) == len(season_months))
    return years, [sum(by_year[y]) / len(season_months) for y in years]


def deseasonalise(months):
    by_month = {}
    for (yr, mo), v in months.items():
        by_month.setdefault(mo, []).append(v)
    clim = {mo: sum(vs) / len(vs) for mo, vs in by_month.items()}
    keys = sorted(months)
    return [yr + (mo - 0.5) / 12 for yr, mo in keys], [months[k] - clim[k[1]] for k in keys]


def mann_kendall(y):
    n = len(y)
    s = sum((y[j] > y[i]) - (y[j] < y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z = (s - 1) / math.sqrt(var_s) if s > 0 else (s + 1) / math.sqrt(var_s) if s < 0 else 0
    return s, z, 2 * (1 - stats.norm.cdf(abs(z)))


def analyse(x, y):
    if len(x) < 4:
        return None
    res = stats.linregress(x, y)
    s, z, p_mk = mann_kendall(y)
    ci = stats.t.ppf(0.975, len(x) - 2) * res.stderr
    sen = stats.theilslopes(y, x)[0]
    return {"slope": res.slope, "intercept": res.intercept, "per_decade": res.slope * 10,
            "ci_decade": ci * 10, "p_t": res.pvalue, "r2": res.rvalue ** 2,
            "mk_s": s, "mk_z": z, "p_mk": p_mk, "sen_decade": sen * 10}


def verdict(a):
    direction = "RISING" if a["slope"] > 0 else "FALLING"
    if a["p_mk"] < 0.05 and a["p_t"] < 0.05:
        return direction, "SIGNIFICANT", "🟢", "Both tests agree (p < 0.05). Scientists would call this a real trend, not random wobble."
    if a["p_mk"] < 0.05 or a["p_t"] < 0.05:
        return direction, "MAYBE", "🟡", "Only one of the two tests says significant. A longer record would settle it."
    return direction, "NOT SIGNIFICANT", "⚪", "The ups and downs are too big to be sure. The trend may be real but needs more years of data."


# ====================================================================== plots
COLORS = {"RISING": "#d62728", "FALLING": "#1f77b4"}

def plot_series(title, unit, months, mode, a, project_to=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    keys = sorted(months)
    xm = [yr + (mo - 0.5) / 12 for yr, mo in keys]
    if mode == "Yearly averages":
        years, ym = yearly_means(months)
        ax.plot(xm, [months[k] for k in keys], color="#cccccc", lw=0.7, label="every month")
        ax.plot(years, ym, "o-", color="#2c3e50", ms=4, label="yearly average")
        xs = years
    else:
        xs, ys = deseasonalise(months)
        ax.plot(xs, ys, color="#2c3e50", lw=0.8, label="monthly anomaly (season removed)")
        ax.axhline(0, color="gray", lw=0.5)
    col = COLORS["RISING" if a["slope"] > 0 else "FALLING"]
    ax.plot(xs, [a["intercept"] + a["slope"] * v for v in xs], "--", color=col, lw=2.2,
            label=f"trend {a['per_decade']:+.3f} {unit} / 10 yr")
    if project_to:
        fx = [xs[-1], project_to]
        ax.plot(fx, [a["intercept"] + a["slope"] * v for v in fx], ":", color=col, lw=2,
                label=f"if it continues to {project_to}")
    ax.set_xlabel("year"); ax.set_ylabel(unit); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_bars(labels, values, unit, title):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cols = [COLORS["RISING"] if v > 0 else COLORS["FALLING"] for v in values]
    ax.bar(labels, values, color=cols)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel(f"{unit} per 10 years"); ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


# ====================================================================== page
st.set_page_config(page_title="Trend Detective", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
h1 {color:#0b3d91;}
.big {font-size:1.6rem; font-weight:700;}
.card {background:#f4f7fb; color:#102a43; border-left:6px solid #0b3d91; padding:0.8rem 1rem; border-radius:8px; margin-bottom:0.6rem;}
.card b {color:#0b3d91;}
</style>""", unsafe_allow_html=True)

st.title("🔎 Trend Detective")
st.caption("Is something on Earth changing? Pick a place, and NASA data tells you what is changing, "
           "how fast, and whether scientists would call it significant. "
           "Team Water Detectives · NASA Space Apps 2026 · Hosur, India")

# ---------------- sidebar
with st.sidebar:
    st.header("1 · Choose a place")
    demo = st.selectbox("Demo places", ["(click the map or search)"] + list(DEMOS))
    city = st.text_input("…or search a city / village", placeholder="e.g. Hosur, Tokyo, Nairobi")
    st.header("2 · Years")
    start, end = st.slider("From / to", 1981, 2025, (1990, 2025))
    mode = st.radio("Analysis mode", ["Yearly averages", "Monthly, season removed"])
    project_to = st.slider("Project the trend into the future (year)", 2026, 2100, 2050)
    st.header("3 · Compare")
    compare = st.checkbox("Compare with a second place")
    if compare:
        demo2 = st.selectbox("Second place", list(DEMOS), index=1)
        lat2, lon2 = DEMOS[demo2]
    st.markdown("---")
    st.caption("Data: NASA POWER (MERRA-2 + satellites). Tests: linear regression t-test, "
               "Mann-Kendall, Theil-Sen slope, 95% confidence interval.")

# ---------------- choose location
if "lat" not in st.session_state:
    st.session_state["lat"], st.session_state["lon"], st.session_state["name"] = 12.74, 77.83, "Hosur, India"
if demo != "(click the map or search)":
    st.session_state["lat"], st.session_state["lon"] = DEMOS[demo]
    st.session_state["name"] = demo
if city:
    g = geocode(city)
    if g:
        st.session_state["lat"], st.session_state["lon"] = round(g[0], 2), round(g[1], 2)
        st.session_state["name"] = g[2].split(",")[0]
    else:
        st.warning("Place not found right now. Check the spelling, or click the map instead.")

lat, lon, name = st.session_state["lat"], st.session_state["lon"], st.session_state["name"]

col_map, col_res = st.columns([1, 1.25])
with col_map:
    m = folium.Map(location=[lat, lon], zoom_start=4, tiles="OpenStreetMap")
    folium.Marker([lat, lon], tooltip=name, icon=folium.Icon(color="red")).add_to(m)
    out = st_folium(m, height=360, width=None, key="map")
    if out and out.get("last_clicked"):
        c = out["last_clicked"]
        if abs(c["lat"] - lat) > 0.01 or abs(c["lng"] - lon) > 0.01:
            st.session_state["lat"], st.session_state["lon"] = round(c["lat"], 2), round(c["lng"], 2)
            st.session_state["name"] = f"{st.session_state['lat']}, {st.session_state['lon']}"
            st.rerun()
    st.markdown(f"<div class='card'>📍 <b>{name}</b> &nbsp; ({lat}, {lon})</div>", unsafe_allow_html=True)
    var_name = st.selectbox("Main variable", list(VARIABLES))
    var, unit, meaning = VARIABLES[var_name]
    st.info(meaning)

# ---------------- download + analyse
try:
    with st.spinner("Asking NASA for the data…"):
        data = fetch_all(lat, lon, start, end)
except Exception as e:
    st.error(f"Could not get NASA data: {e}")
    st.stop()

months = data[var]
years, ym = yearly_means(months)
if mode == "Yearly averages":
    a = analyse(years, ym)
else:
    a = analyse(*deseasonalise(months))
if a is None:
    st.error("Not enough complete years in that range. Widen the year slider.")
    st.stop()
direction, level, icon, why = verdict(a)
total = a["slope"] * (years[-1] - years[0])
future = a["intercept"] + a["slope"] * project_to
if mode != "Yearly averages":
    future_txt = f"{future:+.2f} {unit} above the normal for that month"
else:
    future_txt = f"{future:.2f} {unit}"
hottest = max(zip(ym, years)); coldest = min(zip(ym, years))

with col_res:
    st.pyplot(plot_series(f"{var_name} at {name}", unit, months, mode, a, project_to))
    c1, c2, c3 = st.columns(3)
    c1.metric("Trend", f"{icon} {direction}", f"{a['per_decade']:+.3f} {unit} per 10 yr")
    c2.metric(f"Change {years[0]}→{years[-1]}", f"{total:+.2f} {unit}")
    c3.markdown(f"<div style='font-size:0.85rem;opacity:0.7'>Verdict</div><div class='big'>{icon} {level.title()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'>{why}</div>", unsafe_allow_html=True)
    st.write(f"If this trend continues, {name} would reach **{future_txt}** in **{project_to}** "
             f"(average now {sum(ym)/len(ym):.2f} {unit}). Highest year: **{hottest[1]}** ({hottest[0]:.2f}), "
             f"lowest year: **{coldest[1]}** ({coldest[0]:.2f}).")

# ---------------- tabs with the advanced parts
t1, t2, t3, t4, t5 = st.tabs(["📊 All variables", "🍂 By season", "🗺️ Trend map", "⚖️ Compare", "📄 Report & data"])

with t1:
    st.write("Every variable at this place, same years. Red = rising, blue = falling.")
    rows, labels, vals = [], [], []
    for vn, (code, u, _) in VARIABLES.items():
        ys_, ym_ = yearly_means(data[code])
        aa = analyse(ys_, ym_)
        if aa is None:
            continue
        d, l, ic, _ = verdict(aa)
        rows.append({"Variable": vn, "Unit": u, "Trend per 10 yr": round(aa["per_decade"], 3),
                     "Direction": d, "p (t-test)": round(aa["p_t"], 3), "p (Mann-Kendall)": round(aa["p_mk"], 3),
                     "Verdict": f"{ic} {l}"})
        labels.append(vn); vals.append(aa["per_decade"] / (abs(sum(ym_) / len(ym_)) or 1) * 100)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.pyplot(plot_bars(labels, vals, "% of average", "Trend as percent of the average value, per 10 years"))

with t2:
    st.write(f"Which season of the year is changing most for **{var_name}**?")
    labels, vals, srows = [], [], []
    for sname, smonths in SEASONS.items():
        ys_, ym_ = season_means(months, smonths)
        aa = analyse(ys_, ym_)
        if aa is None:
            continue
        d, l, ic, _ = verdict(aa)
        labels.append(sname); vals.append(aa["per_decade"])
        srows.append({"Season": sname, "Trend per 10 yr": round(aa["per_decade"], 3), "Direction": d,
                      "p (Mann-Kendall)": round(aa["p_mk"], 3), "Verdict": f"{ic} {l}"})
    st.pyplot(plot_bars(labels, vals, unit, f"{var_name}: trend per season"))
    st.dataframe(srows, use_container_width=True, hide_index=True)

with t3:
    st.write("Trend map: the same variable at 9 points around the place (±3°). "
             "Shows if neighbours change the same way, or the opposite way, like the challenge warned.")
    if st.button("Build trend map (9 NASA downloads, ~1 min)"):
        grid = []
        prog = st.progress(0)
        pts = [(lat + dy, lon + dx) for dy in (3, 0, -3) for dx in (-3, 0, 3)]
        for i, (la, lo) in enumerate(pts):
            try:
                dd = fetch_all(round(la, 2), round(lo, 2), start, end)
                ys_, ym_ = yearly_means(dd[var]); aa = analyse(ys_, ym_)
                grid.append((la, lo, aa["per_decade"] if aa else 0, aa["p_mk"] if aa else 1))
            except Exception:
                grid.append((la, lo, 0, 1))
            prog.progress((i + 1) / 9)
        st.session_state["grid"] = (grid, lat, lon, var, unit)
    if "grid" in st.session_state:
        grid, glat, glon, gvar, gunit = st.session_state["grid"]
        mm = folium.Map(location=[glat, glon], zoom_start=5)
        vmax = max(abs(g[2]) for g in grid) or 1
        for la, lo, s_, p in grid:
            colr = COLORS["RISING"] if s_ > 0 else COLORS["FALLING"]
            folium.CircleMarker([la, lo], radius=8 + 22 * abs(s_) / vmax, color=colr, fill=True,
                                fill_opacity=0.8 if p < 0.05 else 0.35,
                                tooltip=f"{s_:+.3f} {gunit}/10yr, p={p:.3f}").add_to(mm)
        st_folium(mm, height=380, width=None, key="trendmap", returned_objects=[])
        st.caption(f"{gvar} trend per decade at 9 points. Bigger circle = stronger trend. "
                   "Solid = significant, faded = not significant. Red = rising, blue = falling.")
        rows = [{"Lat": la, "Lon": lo, f"Trend ({gunit}/10yr)": round(s_, 3), "p (MK)": round(p, 3)} for la, lo, s_, p in grid]
        st.dataframe(rows, use_container_width=True, hide_index=True)

with t4:
    if compare:
        try:
            d2 = fetch_all(lat2, lon2, start, end)[var]
            y2, m2 = yearly_means(d2)
            a2 = analyse(y2, m2) if mode == "Yearly averages" else analyse(*deseasonalise(d2))
            dd2, l2, ic2, w2 = verdict(a2)
            ca, cb = st.columns(2)
            ca.pyplot(plot_series(f"{var_name} at {name}", unit, months, mode, a))
            cb.pyplot(plot_series(f"{var_name} at {demo2}", unit, d2, mode, a2))
            same = a["slope"] * a2["slope"] > 0
            st.markdown(f"<div class='card'><b>{name}</b>: {icon} {direction} {a['per_decade']:+.3f} {unit}/10yr ({level})<br>"
                        f"<b>{demo2}</b>: {ic2} {dd2} {a2['per_decade']:+.3f} {unit}/10yr ({l2})<br>"
                        f"{'Same direction in both places.' if same else '<b>Opposite directions!</b> The same process can push one region up and another down.'}</div>",
                        unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Second place failed: {e}")
    else:
        st.info("Tick 'Compare with a second place' in the sidebar.")

with t5:
    report = io.StringIO()
    report.write(f"TREND DETECTIVE REPORT  ({datetime.date.today()})\n")
    report.write(f"Place: {name} ({lat}, {lon})\nVariable: {var_name} [{var}] in {unit}\n")
    report.write(f"Years: {years[0]}-{years[-1]}  Mode: {mode}\nData: NASA POWER monthly (MERRA-2 + satellite)\n\n")
    report.write(f"Trend: {direction} {a['per_decade']:+.4f} {unit} per decade (95% CI ± {a['ci_decade']:.4f})\n")
    report.write(f"Theil-Sen robust slope: {a['sen_decade']:+.4f} {unit} per decade\n")
    report.write(f"Total change {years[0]}-{years[-1]}: {total:+.3f} {unit}\n")
    report.write(f"t-test p = {a['p_t']:.4f}, R2 = {a['r2']:.3f}\nMann-Kendall S = {a['mk_s']}, Z = {a['mk_z']:.2f}, p = {a['p_mk']:.4f}\n")
    report.write(f"Verdict: {level}. {why}\nProjection: {future_txt} in {project_to} if the trend continues.\n")
    st.code(report.getvalue())
    st.download_button("Download report (.txt)", report.getvalue(), file_name=f"trend_report_{var}_{lat}_{lon}.txt")
    csvbuf = io.StringIO(); w = csv.writer(csvbuf)
    w.writerow(["year", "month", var]); [w.writerow([k[0], k[1], months[k]]) for k in sorted(months)]
    st.download_button("Download the NASA data (.csv)", csvbuf.getvalue(), file_name=f"nasa_power_{var}_{lat}_{lon}.csv")
    with st.expander("How the detective works"):
        st.markdown("""
1. **Get the clue**: one call to NASA POWER brings every monthly value for the place since 1981.
2. **Clean it**: average each year's 12 months (or subtract each month's long-term average to remove the season).
3. **Draw the line**: least-squares linear trend, plus the Theil-Sen slope which ignores outliers.
4. **Test it twice**: a t-test on the slope and the Mann-Kendall test. p < 0.05 in both = SIGNIFICANT.
5. **Explain it**: direction, change per decade, total change, future projection, best and worst years.
        """)

st.markdown("---")
st.caption("Built by Deangelo Reyes (12), team Water Detectives, Hosur, India, for NASA Space Apps 2026. "
           "Data: NASA POWER / MERRA-2 / GPM. Map: OpenStreetMap.")
