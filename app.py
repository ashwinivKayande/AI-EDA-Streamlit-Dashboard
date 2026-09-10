import streamlit as st
import pandas as pd
import plotly.express as px
import ai_insights

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI-Powered EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)



# =========================================================
# PROFESSIONAL UI DESIGN
# =========================================================
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 55%, #e0f2fe 100%);
}

/* Main content width and typography */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1 {
    color: #172554 !important;
    font-weight: 800 !important;
}
h2, h3 {
    color: #1e3a8a !important;
    font-weight: 700 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554 0%, #1e3a8a 55%, #312e81 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 0;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    font-weight: 700;
    padding: 0.55rem 1rem;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.20);
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #4338ca);
    color: white;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 4px 14px rgba(30,64,175,0.08);
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 700;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb;
}

/* Expanders */
div[data-testid="stExpander"] {
    border: 1px solid #dbeafe;
    border-radius: 12px;
    background: rgba(255,255,255,0.75);
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Info / success / warning boxes */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #93c5fd;
    border-radius: 14px;
    background: rgba(239,246,255,0.75);
}

/* Select boxes and inputs */
div[data-baseweb="select"] > div,
input {
    border-radius: 10px !important;
}

/* Horizontal separators */
hr {
    border-color: #bfdbfe !important;
}
</style>
""", unsafe_allow_html=True)




# =========================================================
# HEADER
# =========================================================

st.title("AI-Powered EDA Dashboard")
st.write("Explore your dataset, visualize patterns and generate AI insights.")


# =========================================================
# SIDEBAR - UPLOAD
# =========================================================

st.sidebar.markdown("## Upload Dataset")

st.sidebar.info(
    "Choose a CSV Dataset\n\n"
    "Upload Employee dataset or any CSV file."
)

uploaded_file = st.sidebar.file_uploader(
    "Choose CSV file",
    type=["csv"],
    label_visibility="visible"
)


# =========================================================
# NAVIGATION
# =========================================================

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Visualizations",
        "AI Insights",
        "Report"
    ]
)


# =========================================================
# LOAD DATA
# =========================================================

df = None
filtered_df = None

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        filtered_df = df.copy()

        st.sidebar.success(
            f"{uploaded_file.name}"
        )

    except Exception as e:

        st.sidebar.error(
            f"Error loading CSV: {e}"
        )


# =========================================================
# FILTERS
# =========================================================

if df is not None:

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Filters")

    categorical_cols = (
        df
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    if categorical_cols:

        selected_column = st.sidebar.selectbox(
            "Filter Column",
            ["None"] + categorical_cols
        )

        if selected_column != "None":

            values = (
                df[selected_column]
                .dropna()
                .unique()
                .tolist()
            )

            values = sorted(
                values,
                key=lambda x: str(x)
            )

            selected_values = st.sidebar.multiselect(
                f"Select {selected_column}",
                values,
                default=values
            )

            if selected_values:

                filtered_df = df[
                    df[selected_column].isin(
                        selected_values
                    )
                ]

    else:

        st.sidebar.info(
            "No categorical columns available."
        )


# =========================================================
# PDF REPORT FUNCTION
# =========================================================

def create_pdf_report(data, ai_report=None):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    elements = []

    # =====================================================
    # TITLE
    # =====================================================

    elements.append(
        Paragraph(
            "AI-Powered EDA Report",
            title_style
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    # =====================================================
    # OVERVIEW
    # =====================================================

    elements.append(
        Paragraph(
            "1. Dataset Overview",
            heading_style
        )
    )

    overview = f"""
    Rows: {data.shape[0]}<br/>
    Columns: {data.shape[1]}<br/>
    Missing Values: {int(data.isnull().sum().sum())}<br/>
    Duplicate Rows: {int(data.duplicated().sum())}
    """

    elements.append(
        Paragraph(
            overview,
            normal_style
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    # =====================================================
    # DATA TYPES
    # =====================================================

    elements.append(
        Paragraph(
            "2. Columns and Data Types",
            heading_style
        )
    )

    table_data = [
        ["Column", "Data Type"]
    ]

    for column in data.columns:

        table_data.append(
            [
                str(column),
                str(data[column].dtype)
            ]
        )

    table = Table(table_data)

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ]
        )
    )

    elements.append(table)

    elements.append(
        Spacer(1, 20)
    )

    # =====================================================
    # MISSING VALUES
    # =====================================================

    elements.append(
        Paragraph(
            "3. Missing Values",
            heading_style
        )
    )

    missing = data.isnull().sum()

    for column, value in missing.items():

        elements.append(
            Paragraph(
                f"{column}: {value}",
                normal_style
            )
        )

    elements.append(
        Spacer(1, 20)
    )

    # =====================================================
    # STATISTICS
    # =====================================================

    elements.append(
        Paragraph(
            "4. Statistical Summary",
            heading_style
        )
    )

    numeric_data = data.select_dtypes(
        include="number"
    )

    if not numeric_data.empty:

        summary = (
            numeric_data
            .describe()
            .round(2)
        )

        summary_data = [
            ["Statistic"] +
            list(summary.columns)
        ]

        for index in summary.index:

            summary_data.append(
                [str(index)] +
                [
                    str(v)
                    for v in summary.loc[index]
                ]
            )

        summary_table = Table(
            summary_data,
            repeatRows=1
        )

        summary_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        4
                    )
                ]
            )
        )

        elements.append(summary_table)

    else:

        elements.append(
            Paragraph(
                "No numerical columns found.",
                normal_style
            )
        )

    # =====================================================
    # DATA QUALITY
    # =====================================================

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "5. Data Quality",
            heading_style
        )
    )

    total_cells = (
        data.shape[0] *
        data.shape[1]
    )

    if total_cells > 0:

        missing_percentage = (
            data.isnull().sum().sum()
            / total_cells
        ) * 100

    else:

        missing_percentage = 0

    quality_score = max(
        0,
        round(
            100 -
            missing_percentage
        )
    )

    quality_text = f"""
    Quality Score: {quality_score}/100<br/>
    Missing Percentage: {missing_percentage:.2f}%<br/>
    Duplicate Rows: {int(data.duplicated().sum())}
    """

    elements.append(
        Paragraph(
            quality_text,
            normal_style
        )
    )

    # =====================================================
    # AI REPORT
    # =====================================================

    if ai_report:

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                "6. AI-Generated Analysis",
                heading_style
            )
        )

        for line in ai_report.split("\n"):

            line = line.strip()

            if not line:
                continue

            line = (
                line
                .replace("###", "")
                .replace("##", "")
                .replace("**", "")
            )

            elements.append(
                Paragraph(
                    line,
                    normal_style
                )
            )

            elements.append(
                Spacer(1, 5)
            )

    # =====================================================
    # DATA PREVIEW
    # =====================================================

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "7. Dataset Preview",
            heading_style
        )
    )

    preview = data.head(5)

    preview_data = [
        [
            str(column)
            for column in preview.columns
        ]
    ]

    for _, row in preview.iterrows():

        preview_data.append(
            [
                str(value)
                for value in row
            ]
        )

    preview_table = Table(
        preview_data,
        repeatRows=1
    )

    preview_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    3
                )
            ]
        )
    )

    elements.append(
        preview_table
    )

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "Generated using AI-Powered EDA Dashboard",
            normal_style
        )
    )

    doc.build(elements)

    buffer.seek(0)

    return buffer


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader("Dataset Overview")

    if df is None:

        st.info(
            "Please upload a CSV file from the sidebar."
        )

    else:

        data = filtered_df

        rows = data.shape[0]
        cols = data.shape[1]

        missing = int(
            data.isnull()
            .sum()
            .sum()
        )

        duplicates = int(
            data.duplicated()
            .sum()
        )

        memory = round(
            data.memory_usage(
                deep=True
            ).sum() / 1024**2,
            2
        )

        if len(data) != len(df):

            st.info(
                f"Showing {len(data):,} of "
                f"{len(df):,} rows after filtering."
            )

        # =================================================
        # KPI CARDS
        # =================================================

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Rows",
                f"{rows:,}"
            )

        with c2:

            st.metric(
                "Columns",
                f"{cols:,}"
            )

        with c3:

            st.metric(
                "Missing Values",
                f"{missing:,}"
            )

        with c4:

            st.metric(
                "Duplicate Rows",
                f"{duplicates:,}"
            )

        st.write("")

        # =================================================
        # MEMORY + DATA TYPES
        # =================================================

        left, right = st.columns(2)

        with left:

            st.subheader("Memory Usage")

            st.info(
                f"{memory} MB"
            )

        with right:

            st.subheader("Data Types")

            dtype_df = (
                data.dtypes
                .astype(str)
                .value_counts()
                .reset_index()
            )

            dtype_df.columns = [
                "Data Type",
                "Count"
            ]

            st.dataframe(
                dtype_df,
                use_container_width=True,
                hide_index=True
            )

        st.write("")

        # =================================================
        # DATA PREVIEW
        # =================================================

        st.subheader("Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.write("")

        # =================================================
        # STATISTICS
        # =================================================

        st.subheader("Statistical Summary")

        try:

            st.dataframe(
                data.describe(
                    include="all"
                ).T,
                use_container_width=True
            )

        except Exception:

            st.info(
                "Statistical summary is not available."
            )

        st.write("")

        # =================================================
        # DATA QUALITY
        # =================================================

        st.subheader("Data Quality Score")

        total_cells = rows * cols

        if total_cells > 0:

            missing_percentage = (
                missing /
                total_cells
            ) * 100

        else:

            missing_percentage = 0

        score = max(
            0,
            round(
                100 -
                missing_percentage
            )
        )

        st.progress(
            score / 100
        )

        st.success(
            f"Overall Data Quality: {score}/100"
        )

        st.write("")

        # =================================================
        # MISSING VALUES
        # =================================================

        st.subheader("Missing Values")

        missing_df = (
            data.isnull()
            .sum()
            .reset_index()
        )

        missing_df.columns = [
            "Column",
            "Missing"
        ]

        missing_df = missing_df[
            missing_df["Missing"] > 0
        ]

        if not missing_df.empty:

            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing",
                title="Missing Values by Column"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.success(
                "No missing values found!"
            )


# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "Visualizations":

    st.subheader(
        "Interactive Visualizations"
    )

    if df is None:

        st.info(
            "Please upload a CSV file first."
        )

    else:

        data = filtered_df

        numeric_cols = (
            data
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        categorical_cols = (
            data
            .select_dtypes(
                include=["object", "category"]
            )
            .columns
            .tolist()
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Distribution",
                "Correlation",
                "Relationship",
                "Categorical"
            ]
        )

        # =================================================
        # DISTRIBUTION
        # =================================================

        with tab1:

            if numeric_cols:

                col = st.selectbox(
                    "Select numerical column",
                    numeric_cols
                )

                fig = px.histogram(
                    data,
                    x=col,
                    nbins=30,
                    marginal="box",
                    title=f"Distribution of {col}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                fig2 = px.box(
                    data,
                    y=col,
                    title=f"Box Plot of {col}"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            else:

                st.info(
                    "No numerical columns found."
                )

        # =================================================
        # CORRELATION
        # =================================================

        with tab2:

            if len(numeric_cols) >= 2:

                corr = data[
                    numeric_cols
                ].corr()

                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlation Heatmap"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Need at least 2 numerical columns."
                )

        # =================================================
        # RELATIONSHIP
        # =================================================

        with tab3:

            if len(numeric_cols) >= 2:

                x_col = st.selectbox(
                    "X-axis",
                    numeric_cols,
                    key="x_axis"
                )

                y_col = st.selectbox(
                    "Y-axis",
                    numeric_cols,
                    index=1,
                    key="y_axis"
                )

                fig = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Need at least 2 numerical columns."
                )

        # =================================================
        # CATEGORICAL
        # =================================================

        with tab4:

            if categorical_cols:

                cat_col = st.selectbox(
                    "Select categorical column",
                    categorical_cols
                )

                counts = (
                    data[cat_col]
                    .value_counts()
                    .head(15)
                    .reset_index()
                )

                counts.columns = [
                    cat_col,
                    "Count"
                ]

                fig = px.bar(
                    counts,
                    x=cat_col,
                    y="Count",
                    title=f"Top Categories in {cat_col}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "No categorical columns found."
                )


# =========================================================
# AI INSIGHTS
# =========================================================

elif page == "AI Insights":

    st.subheader(
        "AI-Powered Insights"
    )

    if df is None:

        st.info(
            "Please upload a CSV file first."
        )

    else:

        data = filtered_df

        tab1, tab2 = st.tabs(
            [
                "AI Report",
                "Chat with Dataset"
            ]
        )

        # =================================================
        # AI REPORT
        # =================================================

        with tab1:

            st.markdown(
                "### Generate AI Analysis Report"
            )

            st.write(
                "Generate a professional AI-powered "
                "analysis of your dataset."
            )

            if st.button(
                "Generate AI Insights"
            ):

                with st.spinner(
                    "AI is analyzing your dataset..."
                ):

                    try:

                        insights = (
                            ai_insights
                            .generate_ai_insights(
                                data
                            )
                        )

                        st.success(
                            "AI analysis completed!"
                        )

                        st.markdown(
                            "## AI Analysis Report"
                        )

                        st.markdown(
                            insights
                        )

                        st.download_button(
                            "Download AI Report",
                            data=insights,
                            file_name=(
                                "AI_Insights_Report.txt"
                            ),
                            mime="text/plain"
                        )

                    except Exception as e:

                        st.error(
                            f"AI Error: {e}"
                        )

        # =================================================
        # CHAT
        # =================================================

        with tab2:

            st.markdown(
                "### Ask Questions About Your Dataset"
            )

            question = st.text_input(
                "Ask a question",
                placeholder=(
                    "Example: Which column has "
                    "the most missing values?"
                )
            )

            if st.button(
                "Ask AI"
            ):

                if question.strip():

                    with st.spinner(
                        "AI is thinking..."
                    ):

                        try:

                            answer = (
                                ai_insights
                                .chat_with_dataset(
                                    data,
                                    question
                                )
                            )

                            st.success(
                                "Answer generated!"
                            )

                            st.markdown(
                                "## AI Answer"
                            )

                            st.markdown(
                                answer
                            )

                        except Exception as e:

                            st.error(
                                f"AI Error: {e}"
                            )

                else:

                    st.warning(
                        "Please enter a question."
                    )


# =========================================================
# REPORT
# =========================================================

elif page == "Report":

    st.subheader(
        "Complete AI Report"
    )

    if df is None:

        st.info(
            "Please upload a CSV file first."
        )

    else:

        data = filtered_df

        st.subheader("Generate Complete AI PDF")

        st.write("")

        if st.button(
            "Generate Complete AI PDF"
        ):

            with st.spinner(
                "AI is creating your complete report..."
            ):

                try:

                    ai_report = (
                        ai_insights
                        .generate_ai_insights(
                            data
                        )
                    )

                    pdf_file = (
                        create_pdf_report(
                            data,
                            ai_report
                        )
                    )

                    st.success(
                        "Complete AI PDF generated successfully!"
                    )

                    st.download_button(
                        "Download Complete AI Report",
                        data=pdf_file,
                        file_name=(
                            "AI_Powered_EDA_Report.pdf"
                        ),
                        mime="application/pdf"
                    )

                except Exception as e:

                    st.error(
                        f"Report Error: {e}"
                    )