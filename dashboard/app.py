import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="GeoSales Insight",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GeoSales Insight")
st.markdown("Business Expansion Analysis Dashboard")

# -----------------------------
# Load Data
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "cleaned_sales.csv")
    city = pd.read_csv(DATA_DIR / "city_analysis.csv")

    df["order_date"] = pd.to_datetime(df["order_date"])

    return df, city

df, city = load_data()

df, city = load_data()
st.sidebar.header("Filters")

region = st.sidebar.multiselect(
    "Region",
    sorted(df["region"].unique()),
    default=sorted(df["region"].unique())
)

category = st.sidebar.multiselect(
    "Category",
    sorted(df["category"].unique()),
    default=sorted(df["category"].unique())
)

segment = st.sidebar.multiselect(
    "Segment",
    sorted(df["segment"].unique()),
    default=sorted(df["segment"].unique())
)

filtered_df = df[
    (df["region"].isin(region)) &
    (df["category"].isin(category)) &
    (df["segment"].isin(segment))
]
total_sales = filtered_df["sales"].sum()
total_orders = filtered_df["order_id"].nunique()
customers = filtered_df["customer_id"].nunique()
cities = filtered_df["city"].nunique()
avg_order = filtered_df.groupby("order_id")["sales"].sum().mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Orders", total_orders)
col3.metric("Customers", customers)
col4.metric("Cities", cities)
col5.metric("Avg Order", f"${avg_order:,.2f}")
st.subheader("Monthly Sales Trend")

monthly = (
    filtered_df
    .groupby(filtered_df["order_date"].dt.to_period("M"))
    ["sales"]
    .sum()
    .reset_index()
)

monthly["order_date"] = monthly["order_date"].astype(str)

fig = px.line(
    monthly,
    x="order_date",
    y="sales",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Sales by Region")

region_sales = (
    filtered_df
    .groupby("region")["sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    region_sales,
    x="region",
    y="sales",
    color="region"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Category Sales")

category_sales = (
    filtered_df
    .groupby("category")["sales"]
    .sum()
    .reset_index()
)

fig = px.pie(
    category_sales,
    names="category",
    values="sales",
    hole=0.4
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Top 10 States")

state_sales = (
    filtered_df
    .groupby("state")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    state_sales,
    x="sales",
    y="state",
    orientation="h"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Top 10 Cities")

city_sales = (
    filtered_df
    .groupby("city")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    city_sales,
    x="sales",
    y="city",
    orientation="h"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Top Customers")

customers = (
    filtered_df
    .groupby("customer_name")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    customers,
    x="sales",
    y="customer_name",
    orientation="h"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Business Expansion Recommendations")

top = city.sort_values(
    "Opportunity_Score",
    ascending=False
).head(10)

st.dataframe(top, use_container_width=True)
st.subheader("City Clusters")

fig = px.scatter(
    city,
    x="Customers",
    y="Total_Sales",
    color="Cluster",
    size="Demand_Score",
    hover_name="city"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Download Results")

csv = city.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Business Recommendations",
    csv,
    "business_recommendations.csv",
    "text/csv"
)