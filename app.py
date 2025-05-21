import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import calendar

alt.renderers.set_embed_options(actions=False)

# -------------------------------
# ⚙️ Load data and config
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_parquet("combined_daily_temperature.parquet")

raw_df = load_data().copy()
raw_df["doy"] = raw_df["date"].dt.dayofyear
raw_df = raw_df[(raw_df["date"].dt.year.between(2015, 2024)) & (raw_df["doy"] != 366)]
locations = sorted(raw_df["location"].unique())

cold_shift_locations = ["Fairbanks AK", "CASS LAKE (Bemidji) MN"]
one_shift_windows = [("01-01", "01-31"), ("02-01", "02-27"), ("11-01", "11-30"), ("12-01", "12-31")]

st.title("🌡️ Field Test Temperature Planner")

st.markdown("""
This tool helps you plan field test windows using historical temperature data from **2015–2024**.

For any selected **location** and **month-day range**, it computes:
- 📈 Avg/Std Dev of TMAX and TMIN
- 📦 Bucket % of days (TMAX, TMIN, TAVG)
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

# -------------------------------
# 📍 Location + Date Range
# -------------------------------
st.subheader("🗺️ Select Location and Date Range")

location = st.selectbox("📍 Select location", locations)

col1, col2 = st.columns(2)
anchor_year = 2025  # Dummy year for consistent calendar logic

months = list(range(1, 13))
month_names = [calendar.month_name[m] for m in months]

# Start month/day
start_month = col1.selectbox("📅 Start Month", months, index=0, format_func=lambda m: calendar.month_name[m])
start_day_options = list(range(1, calendar.monthrange(anchor_year, start_month)[1] + 1))
start_day = col1.selectbox("📅 Start Day", start_day_options, index=0)

# End month/day
end_month = col2.selectbox("📅 End Month", months, index=1, format_func=lambda m: calendar.month_name[m])
end_day_options = list(range(1, calendar.monthrange(anchor_year, end_month)[1] + 1))
default_end_index = min(30, len(end_day_options) - 1)
end_day = col2.selectbox("📅 End Day", end_day_options, index=default_end_index)

# Construct calendar range
start_date = datetime(anchor_year, start_month, start_day)
end_date = datetime(anchor_year, end_month, end_day)
if start_date > end_date:
    st.error("⚠️ Start date must be before end date.")
    st.stop()
calendar_range = pd.date_range(start=start_date, end=end_date)

shifts_per_day = st.slider("⚙️ Shifts per day", 1, 3, 2)


# -------------------------------
# ⚙️ Bucket customization
# -------------------------------
with st.expander("🎛 Customize temperature buckets"):
    default_bins = "-999, -20, 0, 20, 40, 60, 80, 999"
    user_input = st.text_input("Enter bucket breakpoints in °F (comma-separated):", value=default_bins)
    try:
        bins_f = sorted([float(x.strip()) for x in user_input.split(",")])
        bins_c = [(f - 32) * 5 / 9 for f in bins_f]
        bin_labels = [f"{int(bins_f[i])}°F to {int(bins_f[i+1])}°F" for i in range(len(bins_f) - 1)]
    except ValueError:
        st.error("Invalid format. Please enter comma-separated numeric values like -999, -20, 0, 80, 999.")
        st.stop()

bin_count = len(bins_c) - 1
bucket_range = range(bin_count)

# -------------------------------
# 🔎 Filter raw data to selected location and dates
# -------------------------------
doy_list = calendar_range.strftime('%j').astype(int)
df = raw_df[(raw_df["location"] == location) & (raw_df["date"].dt.dayofyear.isin(doy_list))].copy()

if df.empty:
    st.error("⚠️ No data found for the selected range.")
    st.stop()

# -------------------------------
# 📦 Bin TMAX/TMIN/TAVG dynamically
# -------------------------------
df["tmax_bin"] = pd.cut(df["tmax_c"], bins=bins_c, labels=range(bin_count), right=False, include_lowest=True)
df["tmin_bin"] = pd.cut(df["tmin_c"], bins=bins_c, labels=range(bin_count), right=False, include_lowest=True)
df["tavg_bin"] = pd.cut(df["tavg_c"], bins=bins_c, labels=range(bin_count), right=False, include_lowest=True)

# One-hot encode
tmax_bins = pd.get_dummies(df["tmax_bin"], prefix="pr_max_bin")
tmin_bins = pd.get_dummies(df["tmin_bin"], prefix="pr_min_bin")
tavg_bins = pd.get_dummies(df["tavg_bin"], prefix="pr_avg_bin")

# Ensure all expected columns exist
for i in bucket_range:
    for prefix, bins in [("pr_max_bin", tmax_bins), ("pr_min_bin", tmin_bins), ("pr_avg_bin", tavg_bins)]:
        col = f"{prefix}_{i}"
        if col not in bins:
            bins[col] = 0

# Combine all one-hot bins into main DataFrame
df = pd.concat([df, tmax_bins, tmin_bins, tavg_bins], axis=1)

# -------------------------------
# 🧮 Workday weighting
# -------------------------------
shift_weights = pd.Series(1.0, index=calendar_range)

if location in cold_shift_locations:
    for start_str, end_str in one_shift_windows:
        for date in calendar_range:
            if f"{date:%m-%d}" >= start_str and f"{date:%m-%d}" <= end_str and date.weekday() < 5:
                shift_weights[date] = 0.5

n_days = len(calendar_range)
n_weekdays = calendar_range.weekday.isin(range(0, 5)).sum()
adjusted_workdays = shift_weights[calendar_range.weekday < 5].sum()
planned_shifts = int(adjusted_workdays * shifts_per_day)

# -------------------------------
# 📊 Bucket % + shift tables
# -------------------------------
def build_summary_table(type_prefix, data_frame):
    cols = [f"{type_prefix}_{i}" for i in bucket_range]
    bucket_vals = (data_frame[cols].mean() * 100).round(2)
    shifts_100 = (bucket_vals / 100 * planned_shifts).round(0).astype(int)
    shifts_90 = (shifts_100 * 0.9).round(0).astype(int)

    df_out = pd.DataFrame({
        "% of Days": bucket_vals,
        "Shifts (100%)": shifts_100,
        "Shifts (90%)": shifts_90
    })

    df_out.index = bin_labels
    total = df_out[["Shifts (100%)", "Shifts (90%)"]].sum(numeric_only=True).to_frame().T
    total.index = ["TOTAL"]
    total["% of Days"] = np.nan
    return pd.concat([df_out, total])

# -------------------------------
# 📍 Summary for selected location
# -------------------------------
st.markdown(f"### 📍 Summary for **{location}**")
st.markdown(f"- **Calendar days**: {n_days}")
st.markdown(f"- **Weekdays**: {n_weekdays}")
if location in cold_shift_locations:
    st.markdown(f"- **Adjusted for 1-shift rules**: {adjusted_workdays:.1f} effective workdays")
st.markdown(f"- **Planned shifts**: {planned_shifts}")

# Individual location tables
df_max = build_summary_table("pr_max_bin", df)
df_min = build_summary_table("pr_min_bin", df)
df_avg = build_summary_table("pr_avg_bin", df)

st.subheader("🔥 TMAX Buckets")
st.dataframe(df_max.style.format({"% of Days": "{:.2f}", "Shifts (100%)": "{:.0f}", "Shifts (90%)": "{:.0f}"}).apply(
    lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index], axis=0))

st.subheader("❄️ TMIN Buckets")
st.dataframe(df_min.style.format({"% of Days": "{:.2f}", "Shifts (100%)": "{:.0f}", "Shifts (90%)": "{:.0f}"}).apply(
    lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index], axis=0))

st.subheader("📊 TAVG Buckets")
st.dataframe(df_avg.style.format({"% of Days": "{:.2f}", "Shifts (100%)": "{:.0f}", "Shifts (90%)": "{:.0f}"}).apply(
    lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index], axis=0))

# -------------------------------
# 📈 Avg Daily Temperatures (°F) Over Selected Window (DOY-averaged)
# -------------------------------
st.subheader("📈 Avg Daily Temperatures (°F) Over Selected Window (2015–2024 Averages)")

# Make sure DOY exists
raw_df = raw_df[raw_df["date"].dt.year.between(2015, 2024)]
raw_df = raw_df[raw_df["doy"] != 366]

doy_list = calendar_range.dayofyear
if "doy" not in raw_df.columns:
    raw_df["doy"] = raw_df["date"].dt.dayofyear
df_plot = raw_df[(raw_df["location"] == location) & (raw_df["doy"].isin(doy_list))]

# Average across years by DOY
avg_by_doy = df_plot.groupby("doy")[["tmax_c", "tmin_c", "tavg_c"]].mean().reset_index()
avg_by_doy["date"] = pd.to_datetime(avg_by_doy["doy"], format="%j")
avg_by_doy["tmax_f"] = avg_by_doy["tmax_c"] * 9/5 + 32
avg_by_doy["tmin_f"] = avg_by_doy["tmin_c"] * 9/5 + 32
avg_by_doy["tavg_f"] = avg_by_doy["tavg_c"] * 9/5 + 32

melted = pd.melt(
    avg_by_doy,
    id_vars=["date"],
    value_vars=["tmax_f", "tmin_f", "tavg_f"],
    var_name="Temperature Type",
    value_name="Degrees (°F)"
)

melted["Temperature Type"] = melted["Temperature Type"].map({
    "tmax_f": "Max Temp (TMAX)",
    "tmin_f": "Min Temp (TMIN)",
    "tavg_f": "Avg Temp (TAVG)"
})

line_chart = alt.Chart(melted).mark_line().encode(
    x=alt.X("date:T", title="Day of Year"),
    y=alt.Y("Degrees (°F)", title="Avg Temp (°F)"),
    color="Temperature Type"
).properties(height=300)

st.altair_chart(line_chart, use_container_width=True)


# -------------------------------
# 🧮 Total Planned Shifts Across All Locations
# -------------------------------
st.subheader("🧮 Total Planned Shifts Across All Locations")

sum_max = pd.Series(dtype="float64")
sum_min = pd.Series(dtype="float64")
sum_avg = pd.Series(dtype="float64")

for loc in locations:
    shift_weights = pd.Series(1.0, index=calendar_range)
    if loc in cold_shift_locations:
        for start_str, end_str in one_shift_windows:
            for date in calendar_range:
                if f"{date:%m-%d}" >= start_str and f"{date:%m-%d}" <= end_str and date.weekday() < 5:
                    shift_weights[date] = 0.5
    adj_days = shift_weights[calendar_range.weekday < 5].sum()
    planned = adj_days * shifts_per_day

    sub = raw_df[(raw_df["location"] == loc) & (raw_df["date"].dt.dayofyear.isin(doy_list))].copy()
    if sub.empty:
        continue

    # Apply dynamic binning for each location
    sub["tmax_bin"] = pd.cut(sub["tmax_c"], bins=bins_c, labels=range(bin_count), right=False, include_lowest=True)
    sub["tmin_bin"] = pd.cut(sub["tmin_c"], bins=bins_c, labels=range(bin_count), right=False, include_lowest=True)
    sub["tavg_bin"] = pd.cut(sub["tavg_c"], bins=bins_c, labels=range(bin_count), right=False, include_lowest=True)

    max_dummies = pd.get_dummies(sub["tmax_bin"], prefix="bin").reindex(columns=[f"bin_{i}" for i in bucket_range], fill_value=0)
    min_dummies = pd.get_dummies(sub["tmin_bin"], prefix="bin").reindex(columns=[f"bin_{i}" for i in bucket_range], fill_value=0)
    avg_dummies = pd.get_dummies(sub["tavg_bin"], prefix="bin").reindex(columns=[f"bin_{i}" for i in bucket_range], fill_value=0)

    sum_max = sum_max.add((max_dummies.mean() * planned), fill_value=0)
    sum_min = sum_min.add((min_dummies.mean() * planned), fill_value=0)
    sum_avg = sum_avg.add((avg_dummies.mean() * planned), fill_value=0)

# Display
def format_total_table(series, label):
    df = pd.DataFrame({
        "Shifts (100%)": series.round(0).astype(int),
        "Shifts (90%)": (series * 0.9).round(0).astype(int)
    })
    df.index = bin_labels
    total = df.sum(numeric_only=True).to_frame().T
    total.index = ["TOTAL"]
    return pd.concat([df, total])

df_sum_max = format_total_table(sum_max, "TMAX")
df_sum_min = format_total_table(sum_min, "TMIN")
df_sum_avg = format_total_table(sum_avg, "TAVG")

st.markdown("### 🔥 TMAX Shift Totals Across All Locations")
st.dataframe(df_sum_max.style.format({"Shifts (100%)": "{:.0f}", "Shifts (90%)": "{:.0f}"}).apply(
    lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index], axis=0))

st.markdown("### ❄️ TMIN Shift Totals Across All Locations")
st.dataframe(df_sum_min.style.format({"Shifts (100%)": "{:.0f}", "Shifts (90%)": "{:.0f}"}).apply(
    lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index], axis=0))

st.markdown("### 📊 TAVG Shift Totals Across All Locations")
st.dataframe(df_sum_avg.style.format({"Shifts (100%)": "{:.0f}", "Shifts (90%)": "{:.0f}"}).apply(
    lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index], axis=0))

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
    shift_weights = pd.Series(1.0, index=calendar_range)
    if loc in cold_shift_locations:
        for start_str, end_str in one_shift_windows:
            for date in calendar_range:
                if f"{date:%m-%d}" >= start_str and f"{date:%m-%d}" <= end_str and date.weekday() < 5:
                    shift_weights[date] = 0.5
    adj_days = shift_weights[calendar_range.weekday < 5].sum()
    planned = adj_days * shifts_per_day

    sub = raw_df[(raw_df["location"] == loc) & (raw_df["date"].dt.dayofyear.isin(doy_list))].copy()
    if sub.empty:
        continue

    # Apply binning for this location
    sub["tmax_bin"] = pd.cut(sub["tmax_c"], bins=bins_c, labels=range(bin_count), right=False, include_lowest=True)
    bucket_dummies = pd.get_dummies(sub["tmax_bin"], prefix="bin").reindex(columns=[f"bin_{i}" for i in bucket_range], fill_value=0)
    bucket_vals = (bucket_dummies.mean() * planned).round(0).astype(int)

    total = int(bucket_vals.sum())
    bucket_vals["Total"] = total
    row = bucket_vals.to_dict()
    row["Location"] = loc
    matrix_rows.append(row)

if matrix_rows:
    df_matrix = pd.DataFrame(matrix_rows).set_index("Location")
    df_matrix.columns = [bin_labels[i] if f"bin_{i}" in df_matrix.columns else c for i, c in enumerate(df_matrix.columns)]
    # Add summary row
    summary_row = df_matrix.sum(numeric_only=True)
    summary_row.name = "TOTAL"
    df_matrix_final = pd.concat([df_matrix, summary_row.to_frame().T])
    st.dataframe(df_matrix_final.style.apply(
        lambda df: ["font-weight: bold" if i == "TOTAL" else "" for i in df.index], axis=0))
else:
    st.info("No data available for selected locations.")

# -------------------------------
# 📉 Temperature Distribution by Location (PDF)
# -------------------------------
st.subheader("📉 Temperature Distribution by Location (PDF)")

temp_type = st.radio("Select temperature type", ["TMAX", "TMIN", "TAVG"], horizontal=True)

selected_locs = st.multiselect(
    "Choose locations to include",
    options=locations,
    default=locations
)

if temp_type == "TMAX":
    temp_column = "tmax_c"
    bins = np.linspace(-60, 120, 100)
    chart_title = "TMAX Distribution by Location (°F)"
elif temp_type == "TMIN":
    temp_column = "tmin_c"
    bins = np.linspace(-60, 100, 100)
    chart_title = "TMIN Distribution by Location (°F)"
else:
    temp_column = "tavg_c"
    bins = np.linspace(-60, 120, 100)
    chart_title = "TAVG Distribution by Location (°F)"

bin_centers = (bins[:-1] + bins[1:]) / 2
pdf_chart_data = []

for loc in selected_locs:
    sub = raw_df[(raw_df["location"] == loc) & (raw_df["date"].dt.dayofyear.isin(doy_list))]
    values = sub[temp_column].dropna() * 9/5 + 32  # Convert to °F

    if values.empty:
        continue

    pdf, _ = np.histogram(values, bins=bins, density=True)
    pdf_chart_data.append(pd.DataFrame({
        "Temperature (°F)": bin_centers,
        "Density": pdf,
        "Location": loc
    }))

if pdf_chart_data:
    pdf_df = pd.concat(pdf_chart_data)

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

