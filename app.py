import streamlit as st
import pandas as pd
import plotly.express as px
from ai_insights import generate_ai_insights

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title{
        font-size:40px;
        font-weight:bold;
        color:#1f77b4;
        text-align:center;
    }
    .sub-title{
        font-size:18px;
        color:gray;
        text-align:center;
    }
    .stApp{
        background: linear-gradient(to right, #f8fbff, #eef5ff);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown('<p class="main-title">AI-Powered EDA Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Interactive Exploratory Data Analysis with Streamlit</p>', unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# --------------------------------------------------
# Main app
# --------------------------------------------------
if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Metrics
    rows, cols = df.shape
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    # Sidebar filter (optional)
    if "Department" in df.columns:
        departments = st.sidebar.multiselect(
            "Filter Department",
            df["Department"].dropna().unique(),
            default=df["Department"].dropna().unique()
        )
        df = df[df["Department"].isin(departments)]

        rows, cols = df.shape
        missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Rows", rows)
    m2.metric("Columns", cols)
    m3.metric("Missing Values", missing)
    m4.metric("Duplicates", duplicates)

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "Overview",
        "Visualizations",
        "AI Report"
    ])

    # --------------------------------------------------
    # Tab 1 - Overview
    # --------------------------------------------------
    with tab1:

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info_df, use_container_width=True)

        st.subheader("Missing Value Analysis")

        missing_df = pd.DataFrame({
            "Missing Count": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
        })

        st.dataframe(missing_df, use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include="all"), use_container_width=True)

    # --------------------------------------------------
    # Tab 2 - Visualizations
    # --------------------------------------------------
    with tab2:

        st.subheader("Interactive Visualizations")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        if numeric_cols:

            selected_num = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            c1, c2 = st.columns(2)

            with c1:
                fig = px.histogram(
                    df,
                    x=selected_num,
                    title=f"Distribution of {selected_num}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.box(
                    df,
                    y=selected_num,
                    title=f"Box Plot of {selected_num}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:

            st.subheader("Correlation Heatmap")

            corr = df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Correlation Matrix"
            )

            st.plotly_chart(fig, use_container_width=True)

        if categorical_cols:

            st.subheader("Categorical Analysis")

            selected_cat = st.selectbox(
                "Select Categorical Column",
                categorical_cols
            )

            counts = df[selected_cat].value_counts().reset_index()
            counts.columns = [selected_cat, "Count"]

            c1, c2 = st.columns(2)

            with c1:
                fig = px.bar(
                    counts,
                    x=selected_cat,
                    y="Count",
                    title=f"{selected_cat} Distribution",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.pie(
                    counts,
                    names=selected_cat,
                    values="Count",
                    title=f"{selected_cat} Share"
                )
                st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:

            st.subheader("Relationship Analysis")

            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", numeric_cols, index=1)

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Tab 3 - AI Report
    # --------------------------------------------------
    with tab3:

        st.subheader("AI-generated Insights")

        summary = f"""
Rows: {rows}
Columns: {cols}

Columns:
{list(df.columns)}

Data Types:
{df.dtypes}

Missing Values:
{df.isnull().sum()}

Statistical Summary:
{df.describe(include='all')}
"""

        if st.button("Generate AI Insights"):

            with st.spinner("AI is analyzing your dataset..."):

                try:

                    insights = generate_ai_insights(summary)

                    st.success("Analysis complete!")

                    st.markdown("### AI Analysis Report")
                    st.markdown(insights)

                    st.download_button(
                        label="Download AI Report",
                        data=insights,
                        file_name="AI_EDA_Report.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("Upload a CSV file from the sidebar to begin analysis.")