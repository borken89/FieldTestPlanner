# 🌡️ Field Test Temperature Planner

This Streamlit app helps reliability and field test engineers identify optimal test windows based on historical temperature data from **2015–2024**.

The tool analyzes daily climate data across North America using public sources from:
- 🇺🇸 **NOAA GHCN-D** ([Find stations](https://www.ncdc.noaa.gov/cdo-web/datatools/findstation))
- 🇨🇦 **Environment and Climate Change Canada** ([Find stations](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html))

---

## 🔍 What the app does

For any selected **test location** and **month-day range**, the app calculates:

- 📈 Average and standard deviation of TMAX, TMIN, and TAVG  
- 📦 Distribution of days across defined temperature buckets (e.g. `80°F–100°F`)  
- 🧮 Estimated number of test shifts per bucket (100% and 90% planning targets)  
- 📊 Temperature probability density functions (PDFs)  
- 🗂️ Matrix-style comparison across multiple test locations

---

## 🗂️ Included Locations (examples)

- Fairbanks, AK
- San Antonio, TX
- New Carlisle, IN
- Yuma, AZ
- Edmonton, AB (Canada)
- Kapuskasing, ON (Canada)

Full station list is included in the app under “📍 View Station Names and IDs.”

---

## 🚀 How to run the app

### Option 1: Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
