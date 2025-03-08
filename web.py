import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import nbimporter
from data_process import plot_crash_simulation

# Set Streamlit page title
st.set_page_config(page_title="Chicago Traffic Accident Analysis", layout="wide")

st.title("🚦 Chicago Traffic Accident Data Analysis")
st.markdown("This webpage presents various data analysis results on Chicago traffic accidents.")

@st.cache_data
def load_data():
    df = pd.read_csv("chicago_accidents_cleaned.csv")
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    return df

df = load_data()

# Display data overview
st.subheader("📊 Data Overview")
st.write(df.head())
# ========== Accident Query Function ==========
st.subheader("🔍 Query Specific Accidents")
crash_id = st.text_input("Enter `Crash_Record_ID` to search for accident details")
if crash_id:
    result = df[df["crash_record_id"].astype(str) == crash_id]
    if result.empty:
        st.warning("No accident record found")
    else:
        st.write(result)

st.markdown("📌 **Data Source**: Chicago Open Data API")
# ========== Accident Map Visualization ==========
st.subheader("🗺️ Traffic Accident Map in Chicago")

# Filter valid latitude and longitude data
df_map = df.dropna(subset=["latitude", "longitude"])

# Create a map centered on downtown Chicago
m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

# Add accident location markers
for _, row in df_map.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.6,
    ).add_to(m)

# Display the map in Streamlit
folium_static(m)

st.markdown("**Note**: The red dots represent accident locations in Chicago.")
# ========== Chicago Accident Clusters ==========
# Filter out rows with missing latitude/longitude
geo_df = df.dropna(subset=["latitude", "longitude"])

# Select the number of clusters (hotspots)
k = 5  # You can adjust this number based on the map

# Standardizing the latitude and longitude for clustering
scaler = StandardScaler()
geo_scaled = scaler.fit_transform(geo_df[["latitude", "longitude"]])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(geo_scaled)
geo_df["cluster"] = kmeans.labels_

# Get cluster centers in original coordinate space
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Create a map centered on Chicago
st.subheader("🚨 High-Risk Accident Hotspots in Chicago")
m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

# Plot accident locations
for _, row in geo_df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.6,
    ).add_to(m)

# Plot cluster centers
for center in cluster_centers:
    folium.Marker(
        location=[center[0], center[1]],
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

# Display map in Streamlit
folium_static(m)

st.markdown("The **red markers** indicate the most high-risk accident zones in Chicago based on clustering analysis.")

# ========== Accident Time Distribution ==========
st.subheader("Accident Time Distribution")

st.markdown("### Hourly Distribution of Accidents")
fig, ax = plt.subplots(figsize=(5, 3))
df["crash_date"].dt.hour.value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
plt.xlabel("Hour of Accident")
plt.ylabel("Number of Accidents")
st.pyplot(fig, use_container_width=False)

# ========== Impact of Lighting Conditions ==========
st.subheader("Impact of Lighting Conditions on Accidents")
fig, ax = plt.subplots(figsize=(5, 3))
df["lighting_condition"].value_counts().plot(kind="bar", ax=ax, color="purple")
plt.xlabel("Lighting Condition")
plt.ylabel("Number of Accidents")
plt.xticks(fontsize=6)
plt.yticks(fontsize=5)
st.pyplot(fig, use_container_width=False)
#========== Average Daily Accidents by Weather Condition ==========
st.subheader("🌦️ Average Daily Accidents by Weather Condition")

# Calculate the number of accidents per day
daily_accidents = df.groupby(["crash_date", "weather_condition"]).size().reset_index(name="accident_count")

# Compute the total number of days for each weather condition
weather_days = daily_accidents.groupby("weather_condition")["crash_date"].nunique()

# Calculate the average number of accidents per day for each weather condition
weather_accidents = df["weather_condition"].value_counts()
avg_accidents_per_day = weather_accidents / weather_days

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
avg_accidents_per_day.sort_values(ascending=False).plot(kind="bar", ax=ax, color="skyblue")
plt.xlabel("Weather Condition")
plt.ylabel("Average Daily Number of Accidents")
plt.title("Average Daily Accidents by Weather Condition")
plt.xticks(rotation=45)
st.pyplot(fig)

# ========== Road Conditions vs Accidents ==========
st.subheader("Relationship Between Road Conditions and Accidents")
fig, ax = plt.subplots(figsize=(5, 3))
df["roadway_surface_cond"].value_counts().plot(kind="bar", ax=ax, color="green")
plt.xlabel("Road Surface Condition")
plt.ylabel("Number of Accidents")
plt.xticks(fontsize=6)
plt.yticks(fontsize=5)
st.pyplot(fig, use_container_width=False)
# ========== Accidents by Traffic Control Device ==========
st.subheader("🚦 Accidents by Traffic Control Device")

# Analyze accident frequency with and without traffic signals
traffic_signal_counts = df.groupby("traffic_control_device").size()

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
traffic_signal_counts.plot(kind="bar", ax=ax, color="darkred")
plt.xlabel("Traffic Control Device")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Traffic Control Device")
plt.xticks(rotation=45)
st.pyplot(fig)
# ========== Top Contributing Factors ==========
st.subheader("⚠️Top Contributing Factors of Accidents")
fig, ax = plt.subplots(figsize=(5, 4))
df["prim_contributory_cause"].value_counts().head(20).plot(kind="bar", ax=ax, color="red")
plt.xlabel("Primary Contributory Cause")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=75,fontsize=3)
plt.yticks(fontsize=5)
st.pyplot(fig, use_container_width=False)

# ========== Crash Type and Trafficway Type Relationship ==========
st.subheader("🚗 Impact of Trafficway Type on Crash Types")
fig, ax = plt.subplots(figsize=(5, 3))
df.groupby(["trafficway_type", "first_crash_type"]).size().unstack().plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
plt.xlabel("Trafficway Type")
plt.ylabel("Number of Accidents")
plt.legend(title="Crash Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
st.pyplot(fig, use_container_width=False)

# ========== Injuries and Fatalities by Crash Type ==========
st.subheader("🚗 Injuries and Fatalities by Crash Type")

# Count the number of injury/fatality accidents for each crash type
injury_counts = df.groupby("first_crash_type")[["injuries_total"]].sum()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
injury_counts.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
plt.xlabel("Crash Type")
plt.ylabel("Number of Injuries/Fatalities")
plt.title("Injuries and Fatalities by Crash Type")
plt.xticks(rotation=45)
st.pyplot(fig)
# ========== Speed vs Accidents ==========
st.subheader("Speed Limit and Accident Frequency")
fig, ax = plt.subplots(figsize=(5, 3))
df.groupby("speed_limit").size().plot(kind="bar", ax=ax, color="teal")
plt.xlabel("Speed Limit (mph)")
plt.ylabel("Number of Accidents")
st.pyplot(fig, use_container_width=False)


