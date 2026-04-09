import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

from src.dashboard.live_pipeline import process_live_prediction
from src.db.supabase_client import get_supabase_client
from src.config import settings


# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Salary Estimate Assistant",
    page_icon="💼",
    layout="wide",
)

FASTAPI_PREDICT_URL = settings.FASTAPI_PREDICT_URL

# ─── Lookup tables ───────────────────────────────────────────────────────────

EXPERIENCE_LABELS = {
    "EN": "Entry-level",
    "MI": "Mid-level",
    "SE": "Senior",
    "EX": "Executive",
}

EMPLOYMENT_LABELS = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance",
}

COMPANY_SIZE_LABELS = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}

REMOTE_LABELS = {
    0: "On-site",
    50: "Hybrid",
    100: "Fully remote",
}

COUNTRY_LABELS = {
    "AE": "United Arab Emirates", "AT": "Austria", "AU": "Australia",
    "BE": "Belgium", "BR": "Brazil", "CA": "Canada", "CH": "Switzerland",
    "CL": "Chile", "CN": "China", "CO": "Colombia", "DE": "Germany",
    "DK": "Denmark", "DZ": "Algeria", "EE": "Estonia", "ES": "Spain",
    "FR": "France", "GB": "United Kingdom", "GR": "Greece", "HN": "Honduras",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IN": "India",
    "IQ": "Iraq", "IR": "Iran", "IT": "Italy", "JP": "Japan",
    "KE": "Kenya", "LU": "Luxembourg", "MD": "Moldova", "MT": "Malta",
    "MX": "Mexico", "MY": "Malaysia", "NG": "Nigeria", "NL": "Netherlands",
    "NZ": "New Zealand", "PK": "Pakistan", "PL": "Poland", "PR": "Puerto Rico",
    "PT": "Portugal", "RO": "Romania", "RU": "Russia", "SG": "Singapore",
    "SI": "Slovenia", "TN": "Tunisia", "TR": "Turkey", "UA": "Ukraine",
    "US": "United States", "VN": "Vietnam",
}

JOB_TITLE_OPTIONS = [
    "3D Computer Vision Researcher", "AI Scientist", "Analytics Engineer",
    "Applied Data Scientist", "Applied Machine Learning Scientist",
    "BI Data Analyst", "Big Data Architect", "Big Data Engineer",
    "Business Data Analyst", "Computer Vision Engineer", "Data Analyst",
    "Data Analytics Engineer", "Data Analytics Lead", "Data Analytics Manager",
    "Data Architect", "Data Engineer", "Data Engineering Manager",
    "Data Scientist", "Director of Data Engineering", "Director of Data Science",
    "ETL Developer", "Finance Data Analyst", "Financial Data Analyst",
    "Head of Data", "Head of Data Science", "Lead Data Analyst",
    "Lead Data Engineer", "Lead Data Scientist", "Lead Machine Learning Engineer",
    "Machine Learning Developer", "Machine Learning Engineer",
    "Machine Learning Infrastructure Engineer", "Machine Learning Manager",
    "Machine Learning Scientist", "Marketing Data Analyst", "ML Engineer",
    "NLP Engineer", "Principal Data Analyst", "Principal Data Engineer",
    "Principal Data Scientist", "Product Data Analyst", "Research Scientist",
    "Staff Data Scientist",
]

# ─── Global styles ───────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        .stApp {
            background: #F8FAFC;
        }

        .block-container {
            max-width: 1150px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        h1, h2, h3, h4 {
            color: #0F172A;
        }

        .hero-banner {
            background: linear-gradient(135deg, #0F172A 0%, #2563EB 100%);
            border-radius: 22px;
            padding: 2.2rem 2.4rem;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 40px rgba(37, 99, 235, 0.16);
        }

        .hero-banner h1 {
            color: white;
            margin: 0 0 0.45rem 0;
            font-size: 2.15rem;
            font-weight: 700;
        }

        .hero-banner p {
            margin: 0;
            font-size: 1.02rem;
            color: rgba(255,255,255,0.85);
            line-height: 1.7;
        }

        .trust-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin: 1rem 0 1.25rem;
        }

        .trust-pill {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 999px;
            padding: 0.55rem 0.9rem;
            font-size: 0.88rem;
            color: #334155;
        }

        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #64748B;
            margin: 1.6rem 0 0.8rem;
        }

        .content-card {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 1.35rem 1.35rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }

        .step-card {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.03);
            margin-bottom: 1rem;
        }

        .step-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.25rem;
        }

        .step-subtitle {
            font-size: 0.9rem;
            color: #64748B;
            margin-bottom: 1rem;
        }

        .result-hero {
            background: linear-gradient(180deg, #EFF6FF 0%, #FFFFFF 100%);
            border: 1px solid #BFDBFE;
            border-radius: 22px;
            padding: 2rem 1.75rem;
            text-align: center;
            margin-top: 1rem;
            box-shadow: 0 18px 40px rgba(37, 99, 235, 0.08);
        }

        .result-eyebrow {
            font-size: 0.8rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #2563EB;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .result-amount {
            font-size: 3rem;
            line-height: 1;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.55rem;
        }

        .result-caption {
            font-size: 1rem;
            color: #475569;
            margin-bottom: 0.7rem;
        }

        .why-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1rem 1rem;
            text-align: left;
            margin-top: 1rem;
        }

        .why-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.35rem;
        }

        .why-text {
            font-size: 0.95rem;
            color: #334155;
            line-height: 1.7;
        }

        .kpi-grid {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 0.75rem;
            margin-bottom: 1.25rem;
        }

        .kpi-card {
            flex: 1;
            min-width: 170px;
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }

        .kpi-label {
            font-size: 0.78rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.3rem;
        }

        .kpi-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0F172A;
        }

        .kpi-sub {
            font-size: 0.82rem;
            color: #94A3B8;
            margin-top: 0.15rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: none;
            background: #2563EB;
            color: white;
            font-weight: 700;
            padding: 0.85rem 1rem;
            font-size: 1rem;
        }

        .stButton > button:hover {
            background: #1D4ED8;
            color: white;
        }

        .small-note {
            color: #64748B;
            font-size: 0.88rem;
            margin-top: 0.5rem;
        }

        .stDataFrame {
            border-radius: 14px;
            overflow: hidden;
        }

        .streamlit-expanderHeader {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0F172A;
        }

        hr {
            border-color: #E2E8F0;
            margin: 1.75rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Supabase helpers ────────────────────────────────────────────────────────

@st.cache_resource
def init_supabase():
    return get_supabase_client()


def fetch_all_runs(supabase) -> pd.DataFrame:
    response = (
        supabase.table("prediction_runs")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return pd.DataFrame(response.data) if response.data else pd.DataFrame()


def fetch_predictions(supabase, run_id: int) -> pd.DataFrame:
    response = (
        supabase.table("predictions")
        .select("*")
        .eq("run_id", run_id)
        .execute()
    )
    return pd.DataFrame(response.data) if response.data else pd.DataFrame()


def fetch_analysis(supabase, run_id: int):
    response = (
        supabase.table("analyses")
        .select("*")
        .eq("run_id", run_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None

def format_runs_for_selectbox(runs_df: pd.DataFrame):
    options = []
    for _, row in runs_df.iterrows():
        label = f"Saved result #{row['id']} · {row['run_name']} · {row['created_at'][:16]}"
        options.append((label, row["id"]))
    return options


def get_country_options() -> list[str]:
    preferred = ["US", "GB", "CA", "DE", "FR", "IN", "ES", "NL", "AU", "JP"]
    remaining = sorted([c for c in COUNTRY_LABELS if c not in preferred])
    return preferred + remaining


# ─── Render helpers ──────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>💼 Salary Estimate Assistant</h1>
            <p>
                Enter a few job details and get a simple salary estimate with a clear explanation.
                Built to make salary insights easy to understand for everyone.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="trust-row">
            <div class="trust-pill">🌍 Based on global salary patterns</div>
            <div class="trust-pill">📈 Compare with saved estimates</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metrics(runs_df: pd.DataFrame, supabase):
    if runs_df.empty:
        return

    total_runs = len(runs_df)

    all_predictions: list[float] = []
    for run_id in runs_df["id"].tolist()[:20]:
        pdf = fetch_predictions(supabase, run_id)
        if not pdf.empty and "predicted_salary_usd" in pdf.columns:
            all_predictions.extend(pdf["predicted_salary_usd"].dropna().tolist())

    avg_salary = sum(all_predictions) / len(all_predictions) if all_predictions else 0
    max_salary = max(all_predictions) if all_predictions else 0

    st.markdown('<div class="section-label">Platform snapshot</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Saved estimates</div>
                <div class="kpi-value">{total_runs}</div>
                <div class="kpi-sub">Prediction sessions stored</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average estimate</div>
                <div class="kpi-value">${avg_salary:,.0f}</div>
                <div class="kpi-sub">Across recent saved results</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Highest estimate</div>
                <div class="kpi-value">${max_salary:,.0f}</div>
                <div class="kpi-sub">Top predicted annual salary</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Predictions made</div>
                <div class="kpi-value">{len(all_predictions)}</div>
                <div class="kpi-sub">Individual salary estimates</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_prediction():
    st.markdown('<div class="section-label">Get a new estimate</div>', unsafe_allow_html=True)

    country_options = get_country_options()
    default_country_idx = country_options.index("US") if "US" in country_options else 0

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">Step 1 · About the role</div>
                <div class="step-subtitle">Choose the job and experience level.</div>
            """,
            unsafe_allow_html=True,
        )
        job_title = st.selectbox(
            "Job title",
            options=JOB_TITLE_OPTIONS,
            index=JOB_TITLE_OPTIONS.index("Data Scientist"),
            key="job_title",
        )
        experience_level = st.selectbox(
            "Experience level",
            options=list(EXPERIENCE_LABELS.keys()),
            format_func=lambda x: EXPERIENCE_LABELS[x],
            key="experience_level",
        )
        work_year = st.selectbox(
            "Work year",
            options=[2020, 2021, 2022],
            index=2,
            key="work_year",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with step2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">Step 2 · Work setup</div>
                <div class="step-subtitle">Describe how the person works.</div>
            """,
            unsafe_allow_html=True,
        )
        employment_type = st.selectbox(
            "Employment type",
            options=list(EMPLOYMENT_LABELS.keys()),
            format_func=lambda x: EMPLOYMENT_LABELS[x],
            key="employment_type",
        )
        remote_ratio = st.selectbox(
            "Work arrangement",
            options=[0, 50, 100],
            format_func=lambda x: REMOTE_LABELS[x],
            index=2,
            key="remote_ratio",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with step3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">Step 3 · Company details</div>
                <div class="step-subtitle">Choose where the employee and company are based.</div>
            """,
            unsafe_allow_html=True,
        )
        employee_residence = st.selectbox(
            "Employee country",
            options=country_options,
            index=default_country_idx,
            format_func=lambda x: f"{COUNTRY_LABELS.get(x, x)} ({x})",
            key="employee_residence",
        )
        company_location = st.selectbox(
            "Company country",
            options=country_options,
            index=default_country_idx,
            format_func=lambda x: f"{COUNTRY_LABELS.get(x, x)} ({x})",
            key="company_location",
        )
        company_size = st.selectbox(
            "Company size",
            options=list(COMPANY_SIZE_LABELS.keys()),
            format_func=lambda x: COMPANY_SIZE_LABELS[x],
            index=1,
            key="company_size",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.button("Get salary estimate")

    st.markdown(
        '<div class="small-note">This estimate is meant to guide expectations, not replace a formal compensation review.</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        payload = {
            "work_year": work_year,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "job_title": job_title,
            "employee_residence": employee_residence,
            "remote_ratio": remote_ratio,
            "company_location": company_location,
            "company_size": company_size,
        }
        try:
            with st.spinner("Creating salary estimate..."):
                result = process_live_prediction(FASTAPI_PREDICT_URL, payload)
            st.session_state["live_result"] = result
            st.session_state["selected_run_id"] = result["run_id"]
            st.rerun()
        except Exception as exc:
            st.error(f"Something went wrong while creating the estimate: {exc}")

    live_result = st.session_state.get("live_result")
    if live_result:
        st.markdown(
            f"""
            <div class="result-hero">
                <div class="result-eyebrow">Estimated annual salary in USD</div>
                <div class="result-amount">${live_result['predicted_salary_usd']:,.0f}</div>
                <div class="result-caption">Based on the selected profile details</div>

                <div class="why-box">
                    <div class="why-title">Why this estimate?</div>
                    <div class="why-text">{live_result['analysis_text']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _apply_chart_style(ax, title: str):
    ax.set_title(title, fontsize=13, fontweight="600", color="#0F172A", pad=12)
    ax.set_facecolor("white")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.tick_params(colors="#64748B", labelsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.8)


def render_single_chart(filtered_df: pd.DataFrame):
    if filtered_df.empty:
        st.info("No data available to display.")
        return

    st.markdown('<div class="section-label">Simple salary view</div>', unsafe_allow_html=True)

    chart_type = st.radio(
        "Choose a simple view",
        options=["By experience level", "Top job titles"],
        horizontal=True,
    )

    if chart_type == "By experience level":
        exp_data = (
            filtered_df.groupby("experience_level")["predicted_salary_usd"]
            .mean()
            .reindex(["EN", "MI", "SE", "EX"])
            .dropna()
        )
        exp_data.index = [EXPERIENCE_LABELS.get(x, x) for x in exp_data.index]

        if not exp_data.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ["#BFDBFE", "#93C5FD", "#60A5FA", "#2563EB"][:len(exp_data)]
            exp_data.plot(kind="bar", ax=ax, color=colors, width=0.55, edgecolor="none")
            _apply_chart_style(ax, "Average salary by experience level")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.info("This view helps non-technical users quickly see how salary changes as experience grows.")

    else:
        job_data = (
            filtered_df.groupby("job_title")["predicted_salary_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )

        if not job_data.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            job_data.sort_values().plot(kind="barh", ax=ax, color="#3B82F6", edgecolor="none")
            ax.set_title("Top 5 job titles by estimated salary", fontsize=13, fontweight="600", color="#0F172A", pad=12)
            ax.set_facecolor("white")
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.spines["left"].set_color("#E2E8F0")
            ax.tick_params(colors="#64748B", labelsize=10)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
            ax.grid(axis="x", color="#F1F5F9", linewidth=0.8)
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.info("This view highlights which job types tend to receive the highest estimates.")


def render_filters(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return predictions_df

    with st.expander("Show advanced filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        exp_opts = sorted(predictions_df["experience_level"].dropna().unique().tolist())
        emp_opts = sorted(predictions_df["employment_type"].dropna().unique().tolist())
        size_opts = sorted(predictions_df["company_size"].dropna().unique().tolist())

        with col1:
            sel_exp = st.multiselect(
                "Experience level",
                options=exp_opts,
                default=exp_opts,
                format_func=lambda x: EXPERIENCE_LABELS.get(x, x),
            )
        with col2:
            sel_emp = st.multiselect(
                "Employment type",
                options=emp_opts,
                default=emp_opts,
                format_func=lambda x: EMPLOYMENT_LABELS.get(x, x),
            )
        with col3:
            sel_size = st.multiselect(
                "Company size",
                options=size_opts,
                default=size_opts,
                format_func=lambda x: COMPANY_SIZE_LABELS.get(x, x),
            )

    mask = (
        predictions_df["experience_level"].isin(sel_exp)
        & predictions_df["employment_type"].isin(sel_emp)
        & predictions_df["company_size"].isin(sel_size)
    )
    return predictions_df[mask]


def render_run_overview(run_row, filtered_df: pd.DataFrame, analysis):
    st.markdown('<div class="section-label">Saved result overview</div>', unsafe_allow_html=True)

    avg_f = filtered_df["predicted_salary_usd"].mean() if not filtered_df.empty else 0
    max_f = filtered_df["predicted_salary_usd"].max() if not filtered_df.empty else 0
    count = len(filtered_df)

    avg_all = analysis["average_salary"] if analysis else avg_f
    max_all = analysis["max_salary"] if analysis else max_f

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Saved result", f"#{run_row['id']}")
    col2.metric("Prediction engine", run_row["model_name"])
    col3.metric("Average salary", f"${avg_all:,.0f}")
    col4.metric("Rows shown", f"{count}")

    st.markdown(
        f"""
        <div class="small-note">
            Highest salary in this saved result: <strong>${max_all:,.0f}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis(analysis):
    st.markdown('<div class="section-label">Why this saved result looks this way</div>', unsafe_allow_html=True)

    text = analysis.get("analysis_text") if analysis else None
    if text:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="why-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No explanation is available for this saved result.")


def render_predictions(filtered_df: pd.DataFrame):
    st.markdown('<div class="section-label">All results</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No predictions match the selected filters.")
        return

    display_columns = [
        "job_title", "experience_level", "employment_type",
        "employee_residence", "company_location", "remote_ratio",
        "company_size", "predicted_salary_usd", "status",
    ]

    df = filtered_df[display_columns].copy()
    df["experience_level"] = df["experience_level"].map(lambda x: EXPERIENCE_LABELS.get(x, x))
    df["employment_type"] = df["employment_type"].map(lambda x: EMPLOYMENT_LABELS.get(x, x))
    df["company_size"] = df["company_size"].map(lambda x: COMPANY_SIZE_LABELS.get(x, x))
    df["employee_residence"] = df["employee_residence"].map(lambda x: COUNTRY_LABELS.get(x, x))
    df["company_location"] = df["company_location"].map(lambda x: COUNTRY_LABELS.get(x, x))
    df["remote_ratio"] = df["remote_ratio"].map(lambda x: REMOTE_LABELS.get(x, f"{x}%"))
    df["predicted_salary_usd"] = df["predicted_salary_usd"].map(lambda x: f"${x:,.0f}")

    df.columns = [
        "Job Title", "Experience", "Employment", "Employee Country",
        "Company Country", "Work Setup", "Company Size", "Estimated Salary", "Status",
    ]

    st.dataframe(df, use_container_width=True, hide_index=True)

def fetch_eda_assets(supabase) -> pd.DataFrame:
    response = (
        supabase.table("eda_assets")
        .select("*")
        .order("created_at", desc=False)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(response.data)

def render_eda_tab(eda_df: pd.DataFrame):
    st.markdown("### Dataset Insights")

    if eda_df.empty:
        st.info("No EDA charts are available yet.")
        return

    for _, row in eda_df.iterrows():
        st.markdown(f"#### {row['title']}")
        if row.get("description"):
            st.caption(row["description"])
        st.image(row["public_url"], use_container_width=True)
        st.markdown("---")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    render_header()

    try:
        supabase = init_supabase()
    except Exception as exc:
        st.error(f"Could not connect to the database: {exc}")
        return

    runs_df = fetch_all_runs(supabase)

    if not runs_df.empty:
        render_summary_metrics(runs_df, supabase)

    st.markdown("---")

    render_live_prediction()

    st.markdown("---")

    st.markdown('<div class="section-label">Explore saved estimates</div>', unsafe_allow_html=True)

    runs_df = fetch_all_runs(supabase)
    if runs_df.empty:
        st.info("No saved estimates yet. Create your first estimate above.")
        return

    run_options = format_runs_for_selectbox(runs_df)
    option_labels = [label for label, _ in run_options]
    run_map = {run_id: label for label, run_id in run_options}

    selected_run_id = st.session_state.get("selected_run_id")
    default_index = 0
    if selected_run_id in run_map:
        default_index = option_labels.index(run_map[selected_run_id])

    selected_label = st.selectbox(
        "Choose a saved estimate",
        options=option_labels,
        index=default_index,
    )
    selected_run_id = dict(run_options)[selected_label]
    selected_run = runs_df[runs_df["id"] == selected_run_id].iloc[0]

    predictions_df = fetch_predictions(supabase, selected_run_id)
    analysis = fetch_analysis(supabase, selected_run_id)

    filtered_df = render_filters(predictions_df)
    render_run_overview(selected_run, filtered_df, analysis)

    eda_df = fetch_eda_assets(supabase)

    tab_story, tab_chart, tab_table = st.tabs(
        ["📝 Explanation", "📊 Simple chart", "📋 Full results"]
    )

    with tab_story:
        render_analysis(analysis)

    with tab_chart:
        render_single_chart(filtered_df)

    with tab_table:
        render_predictions(filtered_df)
                 
    tab_eda = st.tabs(["📚 Dataset insights"])[0]
    with tab_eda:
        render_eda_tab(eda_df) 

if __name__ == "__main__":
    main()