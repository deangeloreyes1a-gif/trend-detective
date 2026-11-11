# Trend Detective - NASA Space Apps 2026 - team Water Detectives (Deangelo Reyes)
# Challenge: Be An Earth System Trend Detective!
# Data: NASA POWER (MERRA-2 model + satellite data), monthly, from 1990 to last year.
# Needs: pip install requests matplotlib

import requests, math, json, os
import matplotlib.pyplot as plt

PLACES = {
    "hosur":     (12.74, 77.83),
    "chennai":   (13.08, 80.27),
    "bengaluru": (12.97, 77.59),
    "delhi":     (28.61, 77.21),
    "mumbai":    (19.08, 72.88),
    "london":    (51.51, -0.13),
    "sahara":    (23.00, 10.00),
    "amazon":    (-3.00, -60.00),
    "arctic":    (75.00, 0.00),
}

VARIABLES = {
    "1": ("T2M",         "Air temperature at 2 m above ground, monthly average (degrees C)", "deg C"),
    "2": ("PRECTOTCORR", "Rainfall per day, monthly average (mm per day)",                   "mm/day"),
    "3": ("GWETROOT",    "Soil wetness in the root zone, 0 = dry, 1 = fully wet",            "0-1"),
    "4": ("RH2M",        "Relative humidity at 2 m (percent)",                               "%"),
    "5": ("WS2M",        "Wind speed at 2 m (metres per second)",                            "m/s"),
    "6": ("ALLSKY_SFC_SW_DWN", "Sunlight energy reaching the ground (kWh per m2 per day)", "kWh/m2/day"),
}

START, END = 1990, 2025


def get_data(var, lat, lon):
    url = ("https://power.larc.nasa.gov/api/temporal/monthly/point"
           f"?parameters={var}&community=AG&longitude={lon}&latitude={lat}"
           f"&start={START}&end={END}&format=JSON")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"][var]
    years, values = [], []
    for key, v in sorted(data.items()):
        yr, mo = int(key[:4]), int(key[4:])
        if mo == 13 or v is None or v == -999:   # 13 = yearly average, skip it
            continue
        years.append(yr + (mo - 0.5) / 12)
        values.append(v)
    return years, values


def yearly_average(years, values):
    """Average each year's 12 months -> removes the seasonal up-down cycle."""
    sums, counts = {}, {}
    for y, v in zip(years, values):
        k = int(y)
        sums[k] = sums.get(k, 0) + v
        counts[k] = counts.get(k, 0) + 1
    ys = sorted(k for k in sums if counts[k] == 12)
    return ys, [sums[k] / 12 for k in ys]


def linear_trend(x, y):
    """Least-squares line y = a + b*x. Returns slope b, intercept a."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx
    a = my - b * mx
    return b, a


def mann_kendall(y):
    """Mann-Kendall trend test. Returns S, Z and p-value (two-sided)."""
    n = len(y)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += (y[j] > y[i]) - (y[j] < y[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return s, z, p


def analyse(place, lat, lon, var, desc, unit):
    years, values = get_data(var, lat, lon)
    ys, yv = yearly_average(years, values)
    slope, intercept = linear_trend(ys, yv)
    s, z, p = mann_kendall(yv)
    per_decade = slope * 10
    direction = "RISING" if slope > 0 else "FALLING"
    significant = p < 0.05

    # ----- graph -----
    plt.figure(figsize=(10, 5))
    plt.plot(years, values, color="lightgray", linewidth=0.8, label="monthly data (NASA POWER)")
    plt.plot(ys, yv, "o-", color="tab:blue", label="yearly average")
    plt.plot(ys, [intercept + slope * x for x in ys], "--", color="red", linewidth=2,
             label=f"trend: {per_decade:+.2f} {unit} per 10 years")
    plt.title(f"{var} at {place.title()} ({lat}, {lon}) {START}-{END}\n"
              f"{direction}, Mann-Kendall p = {p:.3f} -> "
              f"{'SIGNIFICANT' if significant else 'not significant'}")
    plt.xlabel("year"); plt.ylabel(f"{var} ({unit})"); plt.grid(alpha=0.3); plt.legend()
    os.makedirs("results", exist_ok=True)
    png = f"results/{place}_{var}.png"
    plt.tight_layout(); plt.savefig(png, dpi=150); plt.close()

    # ----- report -----
    print()
    print("=" * 60)
    print(f"TREND DETECTIVE REPORT: {var} at {place.title()}")
    print(f"What it is : {desc}")
    print(f"Years      : {ys[0]} to {ys[-1]} ({len(ys)} years, NASA POWER monthly data)")
    print(f"Average    : {sum(yv)/len(yv):.2f} {unit}")
    print(f"Trend      : {direction} by {abs(per_decade):.3f} {unit} every 10 years")
    print(f"Total change over the period: {slope*(ys[-1]-ys[0]):+.2f} {unit}")
    print(f"Mann-Kendall test: S = {s}, Z = {z:.2f}, p = {p:.4f}")
    if significant:
        print("Verdict    : SIGNIFICANT trend (p < 0.05). Scientists would say this change is real,")
        print("             not just random ups and downs.")
    else:
        print("Verdict    : NOT significant yet (p >= 0.05). Maybe a longer record is needed.")
    print(f"Graph saved: {png}")
    print("=" * 60)
    with open("results/report.txt", "a") as f:
        f.write(f"{place},{var},{ys[0]}-{ys[-1]},{per_decade:+.3f} {unit}/10yr,p={p:.4f},"
                f"{'significant' if significant else 'not significant'}\n")


if __name__ == "__main__":
    print("TREND DETECTIVE  -  NASA Space Apps 2026  -  team Water Detectives")
    print("Is something on Earth changing? Let's find out with NASA data.\n")
    print("Places I know:", ", ".join(PLACES))
    place = input("Which place? (or type lat,lon like 12.74,77.83): ").strip().lower()
    if place in PLACES:
        lat, lon = PLACES[place]
    else:
        lat, lon = [float(x) for x in place.split(",")]
        place = f"{lat}_{lon}"
    print("\nWhat do you want to check?")
    for k, (code, desc, unit) in VARIABLES.items():
        print(f"  {k}. {code:18s} {desc}")
    choice = input("Number: ").strip()
    var, desc, unit = VARIABLES[choice]
    print("\nDownloading from NASA POWER ...")
    analyse(place, lat, lon, var, desc, unit)
