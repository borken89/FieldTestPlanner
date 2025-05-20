import streamlit as st
import pandas as pd
from datetime import datetime
import altair as alt
import numpy as np
import calendar

# -------------------------------
# ⚙️ Load data and config
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_parquet("doy_temperature_buckets.parquet")

bucket_df = load_data()

cold_shift_locations = ["Fairbanks AK", "CASS LAKE (Bemidji) MN"]
one_shift_windows = [("01-01", "01-31"), ("02-01", "02-27"), ("11-01", "11-30"), ("12-01", "12-31")]

# Shared bucket label map for all variables
bucket_label_map = {
    "bin0": "< -30°F",
    "bin1": "-30°F to -20°F",
    "bin2": "-20°F to 0°F",
    "bin3": "0°F to 80°F",
    "bin4": "80°F to 100°F",
    "bin5": "100°F to 105°F",
    "bin6": "105°F to 110°F",
    "bin7": "> 110°F"
}

# -------------------------------
# 📘 Title and Inputs
# -------------------------------
st.title("🌡️ Field Test Temperature Planner")

st.markdown("""
This tool helps you plan field test windows using historical temperature data from **2015–2024**.

For any selected **location** and **month-day range**, it computes:
- 📈 Avg/Std Dev of TMAX and TMIN
- 📦 Bucket % of days
- 🧮 Shift planning estimates (100% & 90%)

ℹ️ Actual average daily temperature (**TAVG**) is used where available.  
TAVG is **not available** for **Bellingham**, **Denver**, **Grants Pass**, **Homestead**, and **Yuma** —  
for these, the average is estimated as *(TMAX + TMIN) / 2*.
            
---

📊 **Data sources**  
- 🇺🇸 **United States**: [NOAA GHCN-D](https://www.ncdc.noaa.gov/cdo-web/datatools/findstation)  
- 🇨🇦 **Canada**: [Environment and Climate Change Canada](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html)

Data is collected, cleaned, and processed using Python and stored locally in `.parquet` format for efficient analysis.
""")

with st.expander("📍 View Station Names and IDs"):
    st.markdown("""
### 🇺🇸 United States (NOAA GHCN-D)

| Station Name | GHCN-D ID |
|--------------|-----------|
| Bangor International Airport, ME | `USW00014606`  
| Bellingham International Airport, WA | `USW00024217`  
| Bemidji Minnesota, MN | `USR0000MBEM`  
| San Antonio International Airport, TX | `USW00012921`  
| Denver Water Department, CO | `USC00052223`  
| Fairbanks International Airport, AK | `USW00026411`  
| Fargo Hector International Airport, ND | `USW00014914`  
| Grants Pass, OR | `USC00353445`  
| Homestead Gen Aviation Airport, FL | `USC00084095`  
| McCarran International Airport (Las Vegas), NV | `USW00023169`  
| Chicago O'Hare International Airport, IL | `USW00094846`  
| South Bend Airport (New Carlisle), IN | `USW00014848`  
| Richmond International Airport, VA | `USW00013740`  
| San Diego International Airport, CA | `USW00023188`  
| Topeka ASOS, KS | `USW00013996`  
| Yuma MCAS, AZ | `USW00003145`  

---

### 🇨🇦 Canada (Environment Canada)

| Station Name | Climate ID |
|--------------|------------|
| Kapuskasing CDA, ON | `6073980`  
| Edmonton International CS, AB | `3012206`
    """)


locations = sorted(bucket_df["location"].unique())
location = st.selectbox("📍 Select location", locations)

col1, col2 = st.columns(2)

import calendar

# Month/day picker with dynamic day list
months = list(range(1, 13))
month_names = [calendar.month_name[m] for m in months]
anchor_year = 2025  # Fixed reference year

# -- Start --
start_month = col1.selectbox("📅 Start Month", months, index=5, format_func=lambda m: calendar.month_name[m])
max_start_day = calendar.monthrange(anchor_year, start_month)[1]
start_day = col1.selectbox("📅 Start Day", list(range(1, max_start_day + 1)), index=0)

# -- End --
end_month = col2.selectbox("📅 End Month", months, index=7, format_func=lambda m: calendar.month_name[m])
max_end_day = calendar.monthrange(anchor_year, end_month)[1]
end_day = col2.selectbox("📅 End Day", list(range(1, max_end_day + 1)), index=max_end_day - 1)

# Create datetime objects (now guaranteed valid)
start_date = datetime(anchor_year, start_month, start_day)
end_date = datetime(anchor_year, end_month, end_day)

shifts_per_day = st.slider("⚙️ Shifts per day", 1, 3, 2)

# -------------------------------
# 🗓️ Build calendar using dummy year
# -------------------------------
# Force same year to ensure clean DOY handling
anchor_year = 2025
start_date = datetime(anchor_year, start_month, start_day)
end_date = datetime(anchor_year, end_month, end_day)

calendar = pd.date_range(start=start_date, end=end_date)
shift_weights = pd.Series(1.0, index=calendar)

if location in cold_shift_locations:
    for start_str, end_str in one_shift_windows:
        for date in calendar:
            if f"{date:%m-%d}" >= start_str and f"{date:%m-%d}" <= end_str and date.weekday() < 5:
                shift_weights[date] = 0.5

n_days = len(calendar)
n_weekdays = calendar.weekday.isin(range(0, 5)).sum()
adjusted_workdays = shift_weights[calendar.weekday < 5].sum()
planned_shifts = int(adjusted_workdays * shifts_per_day)


# -------------------------------
# 📊 Subset and summary stats
# -------------------------------
doy_list = calendar.strftime('%j').astype(int)
subset = bucket_df[(bucket_df["location"] == location) & (bucket_df["doy"].isin(doy_list))]

if subset.empty:
    st.error("⚠️ No temperature data found for this location/date range.")
    st.stop()

tmax_cols = sorted([col for col in subset.columns if col.startswith("pr_max_bin_")])
tmin_cols = sorted([col for col in subset.columns if col.startswith("pr_min_bin_")])

mean_max = subset["expected_tmax"].mean()
std_max = subset["expected_tmax"].std()
mean_min = subset["expected_tmin"].mean()
std_min = subset["expected_tmin"].std()

bucket_max = (subset[tmax_cols].mean() * 100).round(2)
bucket_min = (subset[tmin_cols].mean() * 100).round(2)

shifts_max = (bucket_max / 100 * planned_shifts).round(0).astype(int)
shifts_max_90 = (shifts_max * 0.9).round(0).astype(int)

shifts_min = (bucket_min / 100 * planned_shifts).round(0).astype(int)
shifts_min_90 = (shifts_min * 0.9).round(0).astype(int)

# -------------------------------
# 📈 Chart: Daily TMAX, TMIN, and Avg (Fahrenheit)
# -------------------------------

st.subheader("📈 Avg Daily Temperatures (°F) Over Selected Window")

plot_df = subset[["doy", "expected_tmax", "expected_tmin"]].copy()
plot_df["date"] = pd.to_datetime(plot_df["doy"], format="%j").dt.strftime("%b %d")

# Convert to Fahrenheit
plot_df["expected_tmax"] = plot_df["expected_tmax"] * 9/5 + 32
plot_df["expected_tmin"] = plot_df["expected_tmin"] * 9/5 + 32
plot_df["expected_mean"] = (plot_df["expected_tmax"] + plot_df["expected_tmin"]) / 2

plot_df = plot_df.sort_values("doy")

melted = pd.melt(
    plot_df,
    id_vars=["date"],
    value_vars=["expected_tmax", "expected_tmin", "expected_mean"],
    var_name="Temperature Type",
    value_name="Degrees (°F)"
)
melted["Temperature Type"] = melted["Temperature Type"].map({
    "expected_tmax": "Max Temp (TMAX)",
    "expected_tmin": "Min Temp (TMIN)",
    "expected_mean": "Avg Temp (TAVG)"
})

line_chart = alt.Chart(melted).mark_line().encode(
    x=alt.X("date", title="Date", sort=None),
    y=alt.Y("Degrees (°F)", title="Avg Temp (°F)"),
    color="Temperature Type"
).properties(height=300)

st.altair_chart(line_chart, use_container_width=True)

# -------------------------------
# 📍 Location Summary + Tables
# -------------------------------
st.markdown(f"### 📍 Summary for **{location}**")
st.markdown(f"- **Calendar days**: {n_days}")
st.markdown(f"- **Weekdays**: {n_weekdays}")
if location in cold_shift_locations:
    st.markdown(f"- **Adjusted for 1-shift rules**: {adjusted_workdays:.1f} effective workdays")
st.markdown(f"- **Planned shifts**: {planned_shifts}")

df_max = pd.DataFrame({
    "% of Days": bucket_max,
    "Shifts (100%)": shifts_max,
    "Shifts (90%)": shifts_max_90
}).reindex([f"pr_max_bin_{i}" for i in range(8)], fill_value=0)

df_max.index = df_max.index.map(lambda x: bucket_label_map[f"bin{x.split('_')[-1]}"])

# Add total row (for shift columns only)
summary_row = df_max[["Shifts (100%)", "Shifts (90%)"]].sum(numeric_only=True).to_frame().T
summary_row.index = ["TOTAL"]
summary_row["% of Days"] = ""

df_max = pd.concat([df_max, summary_row])

df_min = pd.DataFrame({
    "% of Days": bucket_min,
    "Shifts (100%)": shifts_min,
    "Shifts (90%)": shifts_min_90
}).reindex([f"pr_min_bin_{i}" for i in range(8)], fill_value=0)

df_min.index = df_min.index.map(lambda x: bucket_label_map[f"bin{x.split('_')[-1]}"])

# Add total row (for shift columns only)
summary_row = df_min[["Shifts (100%)", "Shifts (90%)"]].sum(numeric_only=True).to_frame().T
summary_row.index = ["TOTAL"]
summary_row["% of Days"] = ""

df_min = pd.concat([df_min, summary_row])

st.subheader("🔥 TMAX Buckets")
st.dataframe(df_max)

st.subheader("❄️ TMIN Buckets")
st.dataframe(df_min)

# TAVG Buckets
tavg_cols = sorted([col for col in subset.columns if col.startswith("pr_avg_bin_")])
bucket_avg = (subset[tavg_cols].mean() * 100).round(2)
shifts_avg = (bucket_avg / 100 * planned_shifts).round(0).astype(int)
shifts_avg_90 = (shifts_avg * 0.9).round(0).astype(int)

df_avg = pd.DataFrame({
    "% of Days": bucket_avg,
    "Shifts (100%)": shifts_avg,
    "Shifts (90%)": shifts_avg_90
}).reindex([f"pr_avg_bin_{i}" for i in range(8)], fill_value=0)

df_avg.index = df_avg.index.map(lambda x: bucket_label_map[f"bin{x.split('_')[-1]}"])

# Add TOTAL row
summary_row = df_avg[["Shifts (100%)", "Shifts (90%)"]].sum().to_frame().T
summary_row.index = ["TOTAL"]
summary_row["% of Days"] = df_avg["% of Days"].sum().round(2)
df_avg = pd.concat([df_avg, summary_row])

st.subheader("📊 TAVG Buckets")
st.dataframe(
    df_avg.style
        .format({"% of Days": "{:.2f}", "Shifts (100%)": "{:.0f}", "Shifts (90%)": "{:.0f}"})
        .apply(lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index], axis=0)
)


# -------------------------------
# 📊 Total Across All Locations
# -------------------------------
st.subheader("🧮 Total Planned Shifts Across All Locations")

sum_max = pd.Series(dtype="float64")
sum_min = pd.Series(dtype="float64")
sum_avg = pd.Series(dtype="float64")

for loc in locations:
    shift_weights = pd.Series(1.0, index=calendar)
    if loc in cold_shift_locations:
        for start_str, end_str in one_shift_windows:
            for date in calendar:
                if f"{date:%m-%d}" >= start_str and f"{date:%m-%d}" <= end_str and date.weekday() < 5:
                    shift_weights[date] = 0.5
    adj_days = shift_weights[calendar.weekday < 5].sum()
    planned = adj_days * shifts_per_day

    sub = bucket_df[(bucket_df["location"] == loc) & (bucket_df["doy"].isin(doy_list))]
    if not sub.empty:
        bm = (sub[tmax_cols].mean() * 100)
        sm = (bm / 100 * planned).fillna(0)
        sum_max = sum_max.add(sm, fill_value=0)

        bn = (sub[tmin_cols].mean() * 100)
        sn = (bn / 100 * planned).fillna(0)
        sum_min = sum_min.add(sn, fill_value=0)
        ba = (sub[[f"pr_avg_bin_{i}" for i in range(8)]].mean() * 100)
    sa = (ba / 100 * planned).fillna(0)
    sum_avg = sum_avg.add(sa, fill_value=0)


df_sum_max = pd.DataFrame({
    "Shifts (100%)": sum_max.round(0).astype(int),
    "Shifts (90%)": (sum_max * 0.9).round(0).astype(int)
}).reindex([f"pr_max_bin_{i}" for i in range(8)], fill_value=0)

df_sum_max.index = df_sum_max.index.map(lambda x: bucket_label_map[f"bin{x.split('_')[-1]}"])

# Add summary row
total_row = df_sum_max.sum(numeric_only=True).to_frame().T
total_row.index = ["TOTAL"]
df_sum_max = pd.concat([df_sum_max, total_row])

st.markdown("### 🔥 TMAX Shift Totals Across All Locations")
st.dataframe(df_sum_max)

df_sum_min = pd.DataFrame({
    "Shifts (100%)": sum_min.round(0).astype(int),
    "Shifts (90%)": (sum_min * 0.9).round(0).astype(int)
}).reindex([f"pr_min_bin_{i}" for i in range(8)], fill_value=0)

df_sum_min.index = df_sum_min.index.map(lambda x: bucket_label_map[f"bin{x.split('_')[-1]}"])

# Add summary row
total_row = df_sum_min.sum(numeric_only=True).to_frame().T
total_row.index = ["TOTAL"]
df_sum_min = pd.concat([df_sum_min, total_row])

st.markdown("### ❄️ TMIN Shift Totals Across All Locations")
st.dataframe(df_sum_min)

df_sum_avg = pd.DataFrame({
    "Shifts (100%)": sum_avg.round(0).astype(int),
    "Shifts (90%)": (sum_avg * 0.9).round(0).astype(int)
}).reindex([f"pr_avg_bin_{i}" for i in range(8)], fill_value=0)

df_sum_avg.index = df_sum_avg.index.map(lambda x: bucket_label_map[f"bin{x.split('_')[-1]}"])

# Add summary row
total_row = df_sum_avg.sum(numeric_only=True).to_frame().T
total_row.index = ["TOTAL"]
df_sum_avg = pd.concat([df_sum_avg, total_row])

st.markdown("### 📊 TAVG Shift Totals Across All Locations")
st.dataframe(df_sum_avg.style.apply(
    lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index],
    axis=0
))

# -------------------------------
# 🗂️ Multi-Location Bucket Matrix Table
# -------------------------------
st.subheader("📋 Bucketed Shift Counts by Location")

default_locs = ["Fairbanks AK", "Denver CO", "Las Vegas NV", "San Antonio TX", "New Carlisle IN"]
selected_matrix_locs = st.multiselect(
    "Choose locations to include in comparison",
    options=locations,
    default=default_locs
)

matrix_rows = []

for loc in selected_matrix_locs:
    shift_weights = pd.Series(1.0, index=calendar)
    if loc in cold_shift_locations:
        for start_str, end_str in one_shift_windows:
            for date in calendar:
                if f"{date:%m-%d}" >= start_str and f"{date:%m-%d}" <= end_str and date.weekday() < 5:
                    shift_weights[date] = 0.5
    adj_days = shift_weights[calendar.weekday < 5].sum()
    planned = adj_days * shifts_per_day

    sub = bucket_df[(bucket_df["location"] == loc) & (bucket_df["doy"].isin(doy_list))]
    if sub.empty:
        continue

    # Use TMAX buckets (or you could toggle TMIN if needed)
    bucket_cols = [f"pr_max_bin_{i}" for i in range(7)]
    bucket_vals = (sub[bucket_cols].mean() * planned).round(0).astype(int)

    total = int(bucket_vals.sum())
    bucket_vals["Total"] = total
    row = bucket_vals.to_dict()
    row["Location"] = loc
    matrix_rows.append(row)

if matrix_rows:
    df_matrix = pd.DataFrame(matrix_rows).set_index("Location")
    df_matrix.columns = [bucket_label_map.get(f"bin{c.split('_')[-1]}", c) if c != "Total" else "Total" for c in df_matrix.columns]
    # Add summary row
    summary_row = df_matrix.sum(numeric_only=True)
    summary_row.name = "TOTAL"
    df_matrix_final = pd.concat([df_matrix, summary_row.to_frame().T])
    st.dataframe(df_matrix_final)
else:
    st.info("No data available for selected locations.")


# -------------------------------
# 📉 Temperature Distribution Comparison (PDF)
# -------------------------------

st.subheader("📉 Temperature Distribution by Location (PDF)")

# Toggle between TMAX and TMIN
temp_type = st.radio("Select temperature type", ["TMAX", "TMIN", "TAVG"], horizontal=True)

# Select locations to compare
selected_locs = st.multiselect(
    "Choose locations to include",
    options=locations,
    default=locations  # show all by default
)

# Pick bins and label
if temp_type == "TMAX":
    temp_column = "expected_tmax"
    bins = np.linspace(0, 120, 100)
    chart_title = "TMAX Distribution by Location (°F)"
elif temp_type == "TMIN":
    temp_column = "expected_tmin"
    bins = np.linspace(-60, 100, 100)
    chart_title = "TMIN Distribution by Location (°F)"
else:
    temp_column = "expected_tavg"
    bins = np.linspace(-60, 120, 100)
    chart_title = "TAVG Distribution by Location (°F)"


bin_centers = (bins[:-1] + bins[1:]) / 2
pdf_data = []

for loc in selected_locs:
    raw = bucket_df[(bucket_df["location"] == loc) & (bucket_df["doy"].isin(doy_list))]
    values = raw[temp_column].dropna() * 9/5 + 32  # Convert to °F

    if values.empty:
        continue

    pdf, _ = np.histogram(values, bins=bins, density=True)
    pdf_data.append(pd.DataFrame({
        "Temperature (°F)": bin_centers,
        "Density": pdf,
        "Location": loc
    }))

if pdf_data:
    pdf_df = pd.concat(pdf_data)

    pdf_chart = alt.Chart(pdf_df).mark_line().encode(
        x=alt.X("Temperature (°F)", title="Temperature (°F)"),
        y=alt.Y("Density", title="Probability Density"),
        color="Location"
    ).properties(title=chart_title, height=400)

    st.altair_chart(pdf_chart, use_container_width=True)

    # Optional download
    csv = pdf_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"temp_distribution_{temp_type.lower()}.csv",
        mime="text/csv"
    )
else:
    st.warning("No temperature data available for selected locations.")
