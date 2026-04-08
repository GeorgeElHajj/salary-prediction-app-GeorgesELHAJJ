import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.dashboard.live_pipeline import process_live_prediction
from src.db.supabase_client import get_supabase_client


st.set_page_config(
    page_title="Salary Prediction Platform",
    page_icon="💼",
    layout="wide",
)

FASTAPI_PREDICT_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

EXPERIENCE_LABELS = {
    "EN": "Entry-level",
    "MI": "Mid-level",
    "SE": "Senior",
    "EX": "Executive / Expert",
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

COUNTRY_LABELS = {
    "AE": "United Arab Emirates",
    "AT": "Austria",
    "AU": "Australia",
    "BE": "Belgium",
    "BR": "Brazil",
    "CA": "Canada",
    "CH": "Switzerland",
    "CL": "Chile",
    "CN": "China",
    "CO": "Colombia",
    "DE": "Germany",
    "DK": "Denmark",
    "DZ": "Algeria",
    "EE": "Estonia",
    "ES": "Spain",
    "FR": "France",
    "GB": "United Kingdom",
    "GR": "Greece",
    "HN": "Honduras",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IL": "Israel",
    "IN": "India",
    "IQ": "Iraq",
    "IR": "Iran",
    "IT": "Italy",
    "JP": "Japan",
    "KE": "Kenya",
    "LU": "Luxembourg",
    "MD": "Moldova",
    "MT": "Malta",
    "MX": "Mexico",
    "MY": "Malaysia",
    "NG": "Nigeria",
    "NL": "Netherlands",
    "NZ": "New Zealand",
    "PK": "Pakistan",
    "PL": "Poland",
    "PR": "Puerto Rico",
    "PT": "Portugal",
    "RO": "Romania",
    "RU": "Russia",
    "SG": "Singapore",
    "SI": "Slovenia",
    "TN": "Tunisia",
    "TR": "Turkey",
    "UA": "Ukraine",
    "US": "United States",
    "VN": "Vietnam",
}


st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        h1, h2, h3 {
            color: #0f172a;
        }
        .subtle-text {
            color: #475569;
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_all_job_titles():
    import pandas as pd

    df = pd.read_csv("data/raw/ds_salaries.csv")
    return sorted(df["job_title"].dropna().unique().tolist())

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

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(response.data)


def fetch_predictions(supabase, run_id: int) -> pd.DataFrame:
    response = (
        supabase.table("predictions")
        .select("*")
        .eq("run_id", run_id)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(response.data)


def fetch_analysis(supabase, run_id: int):
    response = (
        supabase.table("analyses")
        .select("*")
        .eq("run_id", run_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]


def format_runs_for_selectbox(runs_df: pd.DataFrame):
    options = []
    for _, row in runs_df.iterrows():
        label = f"Run {row['id']} | {row['run_name']} | {row['created_at']}"
        options.append((label, row["id"]))
    return options


def get_country_options(predictions_df: pd.DataFrame) -> list[str]:
    countries = set()

    if "employee_residence" in predictions_df.columns:
        countries.update(predictions_df["employee_residence"].dropna().unique().tolist())

    if "company_location" in predictions_df.columns:
        countries.update(predictions_df["company_location"].dropna().unique().tolist())

    country_list = sorted(code for code in countries if code in COUNTRY_LABELS)

    if not country_list:
        country_list = sorted(COUNTRY_LABELS.keys())

    return country_list


def render_header():
    st.markdown(
        """
        # 💼 Salary Prediction Platform
        <div class="subtle-text">
            Predict salaries, explore salary patterns, and review AI-generated insights in a clean interactive dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("💡 Create a live prediction, save it automatically, and review it instantly as a stored run.")
    st.markdown("---")


def render_live_prediction(predictions_df: pd.DataFrame):
    st.markdown("## ✨ Live Salary Prediction")
    st.markdown("Fill in the details below. The app will predict the salary, generate a short AI explanation, save the result, and add it to the dashboard.")

    country_options = get_country_options(predictions_df)
    default_country_index = country_options.index("US") if "US" in country_options else 0

    job_title_options = load_all_job_titles()

    with st.form("live_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            work_year = st.selectbox(
                "Work Year",
                options=[2020, 2021, 2022],
                index=2,
            )

            experience_level = st.selectbox(
                "Experience Level",
                options=list(EXPERIENCE_LABELS.keys()),
                format_func=lambda x: f"{x} — {EXPERIENCE_LABELS[x]}",
            )

            employment_type = st.selectbox(
                "Employment Type",
                options=list(EMPLOYMENT_LABELS.keys()),
                format_func=lambda x: f"{x} — {EMPLOYMENT_LABELS[x]}",
            )

            job_title = st.selectbox(
                "Job Title",
                options=job_title_options,
            )

        with col2:
            employee_residence = st.selectbox(
                "Employee Residence",
                options=country_options,
                index=default_country_index,
                format_func=lambda x: f"{x} — {COUNTRY_LABELS.get(x, x)}",
            )

            remote_ratio = st.selectbox(
                "Remote Ratio",
                options=[0, 50, 100],
                format_func=lambda x: f"{x}%",
            )

            company_location = st.selectbox(
                "Company Location",
                options=country_options,
                index=default_country_index,
                format_func=lambda x: f"{x} — {COUNTRY_LABELS.get(x, x)}",
            )

            company_size = st.selectbox(
                "Company Size",
                options=list(COMPANY_SIZE_LABELS.keys()),
                format_func=lambda x: f"{x} — {COMPANY_SIZE_LABELS[x]}",
            )

        submitted = st.form_submit_button("Predict Salary")

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
            with st.spinner("Running prediction and generating AI explanation..."):
                result = process_live_prediction(FASTAPI_PREDICT_URL, payload)

            st.session_state["live_result"] = result
            st.session_state["selected_run_id"] = result["run_id"]

        except Exception as exc:
            st.error(f"Live pipeline failed: {exc}")

    live_result = st.session_state.get("live_result")

    if live_result:
        st.success(f"Predicted Salary: ${live_result['predicted_salary_usd']:,.2f}")
        st.caption(f"Model used: {live_result['model_name']}")
        st.info(live_result["analysis_text"])

    st.markdown("---")


def render_overview(run_row, filtered_df: pd.DataFrame, analysis):
    st.markdown("## 📌 Run Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("🆔 Run ID", run_row["id"])
    col2.metric("🤖 Model", run_row["model_name"])
    col3.metric("🕒 Created At", run_row["created_at"])

    st.markdown("## 💰 Summary Metrics")

    if analysis:
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Salary", f"${analysis['average_salary']:,.0f}")
        col2.metric("Max Salary", f"${analysis['max_salary']:,.0f}")
        col3.metric("Min Salary", f"${analysis['min_salary']:,.0f}")
    else:
        st.info("No analysis summary found for this run.")

    if not filtered_df.empty:
        st.markdown("## 📊 Filtered Snapshot")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows Displayed", len(filtered_df))
        col2.metric("Filtered Avg Salary", f"${filtered_df['predicted_salary_usd'].mean():,.0f}")
        col3.metric("Filtered Max Salary", f"${filtered_df['predicted_salary_usd'].max():,.0f}")

    st.markdown("---")


def render_chart(filtered_df: pd.DataFrame):
    st.markdown("## 📈 Salary Trends")

    if filtered_df.empty:
        st.info("No prediction data available to plot.")
        return

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        experience_chart_df = (
            filtered_df.groupby("experience_level")["predicted_salary_usd"]
            .mean()
            .reindex(["EN", "MI", "SE", "EX"])
            .dropna()
        )

        if not experience_chart_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            experience_chart_df.plot(kind="bar", ax=ax)
            ax.set_title("Average Salary by Experience Level")
            ax.set_xlabel("Experience Level")
            ax.set_ylabel("Salary (USD)")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough data to build the experience-level chart.")

    with chart_col_2:
        job_chart_df = (
            filtered_df.groupby("job_title")["predicted_salary_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )

        if not job_chart_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            job_chart_df.sort_values().plot(kind="barh", ax=ax)
            ax.set_title("Top 5 Job Titles by Predicted Salary")
            ax.set_xlabel("Salary (USD)")
            ax.set_ylabel("Job Title")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough data to build the job-title chart.")

    st.markdown("---")


def render_analysis(analysis):
    st.markdown("## 🤖 AI Insights")

    if analysis and analysis.get("analysis_text"):
        st.info(analysis["analysis_text"])
    else:
        st.info("No analysis text found for this run.")

    st.markdown("---")


def render_filters(predictions_df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("## 🎛️ Filter Results")

    if predictions_df.empty:
        st.info("No prediction data available for filtering.")
        return predictions_df

    with st.expander("Show / Hide Filters", expanded=True):
        col1, col2 = st.columns(2)

        experience_options = sorted(predictions_df["experience_level"].dropna().unique().tolist())
        employment_options = sorted(predictions_df["employment_type"].dropna().unique().tolist())
        company_size_options = sorted(predictions_df["company_size"].dropna().unique().tolist())
        job_title_options = sorted(predictions_df["job_title"].dropna().unique().tolist())

        with col1:
            selected_experience = st.multiselect(
                "🎓 Experience Level",
                options=experience_options,
                default=experience_options,
                format_func=lambda x: f"{x} — {EXPERIENCE_LABELS.get(x, x)}",
            )

            selected_company_size = st.multiselect(
                "🏢 Company Size",
                options=company_size_options,
                default=company_size_options,
                format_func=lambda x: f"{x} — {COMPANY_SIZE_LABELS.get(x, x)}",
            )

        with col2:
            selected_employment = st.multiselect(
                "💼 Employment Type",
                options=employment_options,
                default=employment_options,
                format_func=lambda x: f"{x} — {EMPLOYMENT_LABELS.get(x, x)}",
            )

            selected_job_title = st.multiselect(
                "🧠 Job Title",
                options=job_title_options,
                default=job_title_options,
            )

    filtered_df = predictions_df[
        (predictions_df["experience_level"].isin(selected_experience))
        & (predictions_df["employment_type"].isin(selected_employment))
        & (predictions_df["company_size"].isin(selected_company_size))
        & (predictions_df["job_title"].isin(selected_job_title))
    ]

    st.markdown("---")
    return filtered_df


def render_predictions(filtered_df: pd.DataFrame):
    st.markdown("## 🧾 Prediction Results")

    if filtered_df.empty:
        st.warning("No predictions match the selected filters.")
        return

    display_columns = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "remote_ratio",
        "company_location",
        "company_size",
        "predicted_salary_usd",
        "status",
    ]

    display_df = filtered_df[display_columns].copy()

    display_df["experience_level"] = display_df["experience_level"].map(
        lambda x: f"{x} — {EXPERIENCE_LABELS.get(x, x)}"
    )
    display_df["employment_type"] = display_df["employment_type"].map(
        lambda x: f"{x} — {EMPLOYMENT_LABELS.get(x, x)}"
    )
    display_df["company_size"] = display_df["company_size"].map(
        lambda x: f"{x} — {COMPANY_SIZE_LABELS.get(x, x)}"
    )
    display_df["employee_residence"] = display_df["employee_residence"].map(
        lambda x: f"{x} — {COUNTRY_LABELS.get(x, x)}"
    )
    display_df["company_location"] = display_df["company_location"].map(
        lambda x: f"{x} — {COUNTRY_LABELS.get(x, x)}"
    )
    display_df["remote_ratio"] = display_df["remote_ratio"].map(lambda x: f"{x}%")
    display_df["predicted_salary_usd"] = display_df["predicted_salary_usd"].map(
        lambda x: f"${x:,.0f}"
    )

    st.dataframe(display_df, use_container_width=True)


def main():
    render_header()

    try:
        supabase = init_supabase()
    except Exception as exc:
        st.error(f"Failed to connect to Supabase: {exc}")
        return

    runs_df = fetch_all_runs(supabase)

    latest_predictions_df = pd.DataFrame()
    if not runs_df.empty:
        latest_run_id = runs_df.iloc[0]["id"]
        latest_predictions_df = fetch_predictions(supabase, latest_run_id)

    render_live_prediction(latest_predictions_df)

    runs_df = fetch_all_runs(supabase)
    if runs_df.empty:
        st.warning("No prediction runs found in Supabase yet.")
        return

    st.markdown("## 🗂️ Explore Stored Runs")

    run_options = format_runs_for_selectbox(runs_df)

    selected_run_id = st.session_state.get("selected_run_id")
    option_labels = [label for label, _ in run_options]
    run_map = {run_id: label for label, run_id in run_options}

    default_index = 0
    if selected_run_id in run_map:
        default_index = option_labels.index(run_map[selected_run_id])

    selected_label = st.selectbox(
        "Select a prediction run",
        options=option_labels,
        index=default_index,
    )

    selected_run_id = dict(run_options)[selected_label]
    selected_run = runs_df[runs_df["id"] == selected_run_id].iloc[0]

    predictions_df = fetch_predictions(supabase, selected_run_id)
    analysis = fetch_analysis(supabase, selected_run_id)

    filtered_df = render_filters(predictions_df)

    render_overview(selected_run, filtered_df, analysis)
    render_chart(filtered_df)
    render_analysis(analysis)
    render_predictions(filtered_df)


if __name__ == "__main__":
    main()