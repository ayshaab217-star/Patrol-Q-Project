import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics",
    page_icon="🚔",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.sidebar.title("🚔 PATROLIQ")
st.sidebar.caption("Smart Safety Analytics Platform")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    possible_files = [
        "PatrolIQ_cleaned.csv",
        "PatrolIQ_Final_Clustering_Dataset.csv",
        "PatrolIQ_ML_Features.csv"
    ]

    for file in possible_files:

        if os.path.exists(file):

            df = pd.read_csv(file)

            # Standardize column names
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("-", "_")
            )

            return df, file

    return None, None


df, loaded_file = load_data()

# ============================================================
# CHECK DATA
# ============================================================

if df is None:

    st.error(
        "❌ Dataset not found. Please make sure "
        "PatrolIQ_cleaned.csv is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# SIDEBAR DATA INFO
# ============================================================

st.sidebar.success(f"Loaded: {loaded_file}")

st.sidebar.write(
    f"📊 Records: **{len(df):,}**"
)

st.sidebar.write(
    f"📋 Columns: **{len(df.columns)}**"
)

# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Dashboard",
        "📊 Crime Overview",
        "🕐 Temporal Analysis",
        "🗺️ Geographic Analysis",
        "🚨 Crime Hotspots",
        "🤖 Clustering",
        "📉 PCA Analysis",
        "🔬 t-SNE Analysis",
        "🧪 MLflow Experiments",
        "ℹ️ About Project"
    ]
)

# ============================================================
# HELPER FUNCTION
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        name = name.lower()

        if name in df.columns:
            return name

    return None


# Find important columns automatically

crime_col = find_column([
    "primary_type",
    "crime_type",
    "primarytype"
])

date_col = find_column([
    "date",
    "datetime",
    "crime_date"
])

hour_col = find_column([
    "hour"
])

day_col = find_column([
    "day_of_week",
    "day"
])

month_col = find_column([
    "month"
])

district_col = find_column([
    "district"
])

latitude_col = find_column([
    "latitude",
    "lat"
])

longitude_col = find_column([
    "longitude",
    "lon",
    "lng"
])

arrest_col = find_column([
    "arrest"
])

domestic_col = find_column([
    "domestic"
])


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    st.title("🏠 PatrolIQ Executive Dashboard")

    st.markdown(
        "### Smart Safety Analytics Platform for Chicago Crime Intelligence"
    )

    st.divider()

    # ---------------- KPI CARDS ----------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Crime Records",
            f"{len(df):,}"
        )

    with col2:

        if crime_col:
            st.metric(
                "Crime Types",
                df[crime_col].nunique()
            )
        else:
            st.metric("Crime Types", "N/A")

    with col3:

        if district_col:
            st.metric(
                "Districts",
                df[district_col].nunique()
            )
        else:
            st.metric("Districts", "N/A")

    with col4:

        if arrest_col:

            arrest_rate = (
                pd.to_numeric(
                    df[arrest_col],
                    errors="coerce"
                )
                .mean()
                * 100
            )

            st.metric(
                "Arrest Rate",
                f"{arrest_rate:.2f}%"
            )

        else:
            st.metric("Arrest Rate", "N/A")

    st.divider()

    # ---------------- TOP CRIMES ----------------

    if crime_col:

        st.subheader("🔝 Top Crime Types")

        crime_counts = (
            df[crime_col]
            .value_counts()
            .head(10)
            .reset_index()
        )

        crime_counts.columns = [
            "Crime Type",
            "Count"
        ]

        fig = px.bar(
            crime_counts,
            x="Count",
            y="Crime Type",
            orientation="h",
            title="Top 10 Crime Types"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- DISTRICT ----------------

    if district_col:

        st.subheader("🏙️ Crimes by District")

        district_counts = (
            df[district_col]
            .value_counts()
            .reset_index()
        )

        district_counts.columns = [
            "District",
            "Crime Count"
        ]

        fig = px.bar(
            district_counts,
            x="District",
            y="Crime Count",
            title="Crime Distribution by District"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# CRIME OVERVIEW
# ============================================================

elif page == "📊 Crime Overview":

    st.title("📊 Crime Overview")

    if not crime_col:

        st.warning(
            "Primary crime type column was not found."
        )

        st.write("Available columns:")

        st.write(df.columns.tolist())

        st.stop()

    # ---------------- FILTER ----------------

    crime_types = sorted(
        df[crime_col]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_crime = st.selectbox(
        "Select Crime Type",
        ["All"] + crime_types
    )

    if selected_crime != "All":

        filtered_df = df[
            df[crime_col].astype(str) == selected_crime
        ]

    else:

        filtered_df = df

    st.metric(
        "Selected Crime Records",
        f"{len(filtered_df):,}"
    )

    # ---------------- CRIME DISTRIBUTION ----------------

    counts = (
        filtered_df[crime_col]
        .value_counts()
        .head(15)
        .reset_index()
    )

    counts.columns = [
        "Crime Type",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="Crime Type",
        y="Count",
        title="Crime Type Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

elif page == "🕐 Temporal Analysis":

    st.title("🕐 Temporal Crime Analysis")

    # Create time features if they don't already exist

    temp_df = df.copy()

    if date_col:

        temp_df[date_col] = pd.to_datetime(
            temp_df[date_col],
            errors="coerce"
        )

        if hour_col is None:

            temp_df["hour"] = (
                temp_df[date_col]
                .dt.hour
            )

            hour_col = "hour"

        if month_col is None:

            temp_df["month"] = (
                temp_df[date_col]
                .dt.month
            )

            month_col = "month"

        if day_col is None:

            temp_df["day_of_week"] = (
                temp_df[date_col]
                .dt.day_name()
            )

            day_col = "day_of_week"

    # ---------------- HOURLY ----------------

    if hour_col:

        st.subheader("⏰ Crime by Hour")

        hourly = (
            temp_df[hour_col]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        hourly.columns = [
            "Hour",
            "Crime Count"
        ]

        fig = px.line(
            hourly,
            x="Hour",
            y="Crime Count",
            markers=True,
            title="Crime Occurrences by Hour"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- DAY ----------------

    if day_col:

        st.subheader("📅 Crime by Day of Week")

        daily = (
            temp_df[day_col]
            .value_counts()
            .reset_index()
        )

        daily.columns = [
            "Day",
            "Crime Count"
        ]

        fig = px.bar(
            daily,
            x="Day",
            y="Crime Count",
            title="Crime by Day of Week"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- MONTH ----------------

    if month_col:

        st.subheader("📆 Crime by Month")

        monthly = (
            temp_df[month_col]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        monthly.columns = [
            "Month",
            "Crime Count"
        ]

        fig = px.bar(
            monthly,
            x="Month",
            y="Crime Count",
            title="Crime by Month"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# GEOGRAPHIC ANALYSIS
# ============================================================

elif page == "🗺️ Geographic Analysis":

    st.title("🗺️ Geographic Crime Analysis")

    if latitude_col and longitude_col:

        map_df = df[
            [latitude_col, longitude_col]
        ].copy()

        map_df[latitude_col] = pd.to_numeric(
            map_df[latitude_col],
            errors="coerce"
        )

        map_df[longitude_col] = pd.to_numeric(
            map_df[longitude_col],
            errors="coerce"
        )

        map_df = map_df.dropna()

        # Limit map points for performance

        if len(map_df) > 10000:

            map_df = map_df.sample(
                10000,
                random_state=42
            )

        map_df = map_df.rename(
            columns={
                latitude_col: "lat",
                longitude_col: "lon"
            }
        )

        st.map(
            map_df[
                ["lat", "lon"]
            ]
        )

        st.info(
            "Map displays a representative sample of crime locations."
        )

    else:

        st.warning(
            "Latitude and longitude columns were not found."
        )


# ============================================================
# CRIME HOTSPOTS
# ============================================================

elif page == "🚨 Crime Hotspots":

    st.title("🚨 Crime Hotspot Analysis")

    st.markdown(
        """
        This page visualizes geographic crime concentration
        and hotspot clustering results.
        """
    )

    hotspot_file = "PatrolIQ_KMeans_Results.csv"

    if os.path.exists(hotspot_file):

        hotspot_df = pd.read_csv(
            hotspot_file
        )

        hotspot_df.columns = (
            hotspot_df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        st.subheader(
            "K-Means Geographic Hotspots"
        )

        st.dataframe(
            hotspot_df.head(100),
            use_container_width=True
        )

        lat = find_column([
            "latitude"
        ])

        lon = find_column([
            "longitude"
        ])

    else:

        st.warning(
            "K-Means result file not found."
        )


# ============================================================
# CLUSTERING
# ============================================================

elif page == "🤖 Clustering":

    st.title("🤖 Clustering Analysis")

    st.markdown(
        "Comparison of unsupervised clustering algorithms."
    )

    tabs = st.tabs([
        "K-Means",
        "DBSCAN",
        "Hierarchical"
    ])

    with tabs[0]:

        st.subheader("K-Means Clustering")

        file = "PatrolIQ_KMeans_Results.csv"

        if os.path.exists(file):

            km = pd.read_csv(file)

            st.dataframe(
                km.head(100),
                use_container_width=True
            )

            st.success(
                f"K-Means dataset loaded: {len(km):,} rows"
            )

        else:

            st.warning(
                "K-Means results file not found."
            )

    with tabs[1]:
        st.subheader("DBSCAN Clustering")
        st.write("DBSCAN Parameter Comparison")

        dbscan_params = pd.DataFrame({
            "eps": [1.0, 1.2, 1.5, 2.0],
            "Clusters": [67, 41, 10, 9],
            "Noise (%)": [51.92, 24.44, 5.01, 0.76]
            })

        st.dataframe(
            dbscan_params,
            use_container_width=True
            )

        fig = px.bar(
            dbscan_params,
            x="eps",
            y="Noise (%)",
            title="DBSCAN Noise Percentage by eps"
            )

        st.plotly_chart(
            fig,
            use_container_width=True
            )

        st.write("Final DBSCAN Cluster Distribution")

        dbscan_clusters = pd.DataFrame({
        "Cluster": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Records": [501, 5105, 1550, 2046, 563, 36, 137, 33, 10, 8, 11]
        })

        st.dataframe(
            dbscan_clusters,
            use_container_width=True
            )

    with tabs[2]:
        st.subheader("Hierarchical Clustering")
        hierarchical_results = pd.DataFrame({
        "Cluster": [0, 1, 2, 3, 4],
        "Records": [3628, 3068, 1299, 928, 1077]
        })

        st.write("Hierarchical Cluster Distribution")

        st.dataframe(
            hierarchical_results,
            use_container_width=True
            )
        fig = px.bar(
            hierarchical_results,
            x="Cluster",
            y="Records",
            title="Hierarchical Cluster Distribution"
            )

        st.plotly_chart(
            fig,
            use_container_width=True
            )

        st.subheader("Hierarchical Clustering Dendrogram")

        if os.path.exists("PatrolIQ_Hierarchical_Dendrogram.png"):
            st.image(
            "PatrolIQ_Hierarchical_Dendrogram.png",
            use_container_width=True
            )
        else:
            st.warning(
            "Hierarchical dendrogram image not found."
            )


# ============================================================
# PCA
# ============================================================

elif page == "📉 PCA Analysis":

    st.title("📉 PCA — Principal Component Analysis")

    file = "PatrolIQ_PCA_Results.csv"

    if os.path.exists(file):

        pca_df = pd.read_csv(file)

        st.subheader(
            "PCA Results"
        )

        st.dataframe(
            pca_df.head(100),
            use_container_width=True
        )

        st.success(
            f"PCA results loaded: {len(pca_df):,} rows"
        )

    else:

        st.warning(
            "PatrolIQ_PCA_Results.csv not found."
        )


# ============================================================
# t-SNE
# ============================================================

elif page == "🔬 t-SNE Analysis":

    st.title("🔬 t-SNE Crime Pattern Visualization")

    file = "PatrolIQ_tSNE_Results.csv"

    if os.path.exists(file):

        tsne_df = pd.read_csv(file)

        st.subheader(
            "t-SNE 2D Visualization"
        )

        st.dataframe(
            tsne_df.head(100),
            use_container_width=True
        )

        # Automatically detect t-SNE columns

        x_col = "TSNE1"
        y_col = "TSNE2"

        if x_col and y_col:

            fig = px.scatter(
                tsne_df,
                x=x_col,
                y=y_col,
                title="t-SNE Crime Pattern Clusters"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "PatrolIQ_tSNE_Results.csv not found."
        )


# ============================================================
# MLFLOW
# ============================================================

elif page == "🧪 MLflow Experiments":

    st.title("🧪 MLflow Experiment Tracking")

    st.markdown(
        """
        This section will display model experiments,
        parameters and evaluation metrics.
        """
    )

    st.info(
        "Connect this page to your MLflow tracking results "
        "after the MLflow experiments are completed."
    )

    st.subheader("Expected Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
        "Best Silhouette Score",
        "0.139"
    )

    with col2:
        st.metric(
            "Davies-Bouldin Index",
            "—"
        )

    with col3:
        st.metric(
            "Explained Variance",
            "—"
        )

        st.subheader("MLflow Runs")
        mlflow_results = pd.DataFrame({
            "Algorithm": ["K-Means", "DBSCAN", "K-Means"],
            "Status": ["FINISHED", "FINISHED", "FINISHED"],
            "Silhouette Score": [0.139, 0.026, 0.139]
            })

        st.dataframe(
            mlflow_results,
            use_container_width=True
            )

# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About PatrolIQ")

    st.markdown(
        """
        ## 🚔 PatrolIQ — Smart Safety Analytics Platform

        PatrolIQ is an urban safety analytics platform designed
        to analyze Chicago crime patterns using data analysis
        and unsupervised machine learning.

        ### Technologies

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Plotly
        - Streamlit
        - MLflow

        ### Machine Learning

        - K-Means Clustering
        - DBSCAN
        - Hierarchical Clustering
        - PCA
        - t-SNE

        ### Objective

        Identify geographic crime hotspots, temporal crime
        patterns and meaningful clusters that can support
        data-driven public safety planning.
        """
    )

    st.success(
        "PatrolIQ Smart Safety Analytics Platform 🚔"
    )