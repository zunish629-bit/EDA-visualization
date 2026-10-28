from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DATA_FILE = Path(__file__).parent / "ecommerce_sales_data.csv"
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
COLORS = ["#0F766E", "#2563EB", "#D97706", "#7C3AED", "#DB2777", "#64748B"]
METRICS = {
    "Revenue": "Revenue",
    "Profit": "Profit",
    "Units sold": "Units_Sold",
    "Ad spend": "Ad_Spend",
    "Customer rating": "Customer_Rating",
}


st.set_page_config(
    page_title="Commerce intelligence",
    page_icon=":material/monitoring:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes | None = None) -> pd.DataFrame:
    """Load the bundled dataset or a user-provided CSV and prepare dates."""
    if file_bytes is None:
        frame = pd.read_csv(DATA_FILE)
    else:
        from io import BytesIO

        frame = pd.read_csv(BytesIO(file_bytes))
    required = {
        "Year", "Month", "Category", "Region", "Units_Sold", "Revenue",
        "Profit", "Customer_Rating", "Ad_Spend",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    frame = frame.drop_duplicates().copy()
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame = frame.dropna(subset=["Year", "Month"])
    frame["Year"] = frame["Year"].astype(int)
    frame["date"] = pd.to_datetime(
        frame["Year"].astype(str) + " " + frame["Month"].astype(str),
        format="%Y %B", errors="coerce",
    )
    frame = frame.dropna(subset=["date"])
    frame["Month"] = pd.Categorical(frame["Month"], categories=MONTH_ORDER, ordered=True)
    return frame.sort_values("date").reset_index(drop=True)


def money(value: float) -> str:
    return f"${value:,.0f}"


def chart_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(strokeOpacity=0).configure_axis(
        labelColor="#64748B", titleColor="#475569", gridColor="#E8EDF3",
        domainColor="#CBD5E1", tickColor="#CBD5E1", labelFontSize=11,
        titleFontSize=12,
    ).configure_legend(labelColor="#475569", titleColor="#475569")


st.title("Commerce intelligence", icon=":material/monitoring:")
st.caption("A decision-ready view of sales, margins, demand, and customer signals · 2020–2024")

with st.sidebar:
    st.header("Explore the data", icon=":material/tune:")
    uploaded = st.file_uploader("Use another CSV", type=["csv"], help="Your CSV must include the nine columns in the bundled data.")
    try:
        df = load_data(uploaded.getvalue() if uploaded else None)
    except Exception as error:
        st.error(str(error), icon=":material/error:")
        st.stop()

    years = sorted(df["Year"].unique().tolist())
    categories = sorted(df["Category"].dropna().unique().tolist())
    regions = sorted(df["Region"].dropna().unique().tolist())
    selected_years = st.multiselect("Year", years, default=years)
    selected_categories = st.multiselect("Category", categories, default=categories)
    selected_regions = st.multiselect("Region", regions, default=regions)
    min_rating, max_rating = float(df.Customer_Rating.min()), float(df.Customer_Rating.max())
    rating_range = st.slider("Customer rating", min_rating, max_rating, (min_rating, max_rating), step=0.1)
    measure = st.selectbox("Trend measure", list(METRICS), index=0)
    st.caption(f"{len(df):,} distinct transactions · {df['date'].min():%b %Y} — {df['date'].max():%b %Y}")

filtered = df[
    df["Year"].isin(selected_years)
    & df["Category"].isin(selected_categories)
    & df["Region"].isin(selected_regions)
    & df["Customer_Rating"].between(rating_range[0], rating_range[1])
].copy()

if filtered.empty:
    st.warning("No transactions match these filters. Broaden one or more filters to continue.", icon=":material/filter_alt_off:")
    st.stop()

revenue = filtered["Revenue"].sum()
profit = filtered["Profit"].sum()
units = filtered["Units_Sold"].sum()
margin = profit / revenue if revenue else 0
avg_rating = filtered["Customer_Rating"].mean()

with st.container(horizontal=True):
    st.metric("Revenue", money(revenue), border=True)
    st.metric("Profit", money(profit), f"{margin:.1%} margin", border=True)
    st.metric("Units sold", f"{units:,.0f}", f"{len(filtered):,} transactions", border=True)
    st.metric("Average rating", f"{avg_rating:.2f} / 5", border=True)

# Notebook finding, recalculated in the current filtered slice.
cat_totals = filtered.groupby("Category", observed=True).agg(
    Revenue=("Revenue", "sum"), Profit=("Profit", "sum"),
    Transactions=("Revenue", "size"), Rating=("Customer_Rating", "mean"),
).sort_values("Revenue", ascending=False)
region_totals = filtered.groupby("Region").agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"))
best_category = cat_totals["Revenue"].idxmax()
best_profit_category = cat_totals["Profit"].idxmax()
best_region = region_totals["Revenue"].idxmax()
st.info(
    f"**Readout:** {best_category} leads revenue at {money(cat_totals.loc[best_category, 'Revenue'])}; "
    f"{best_profit_category} leads profit at {money(cat_totals.loc[best_profit_category, 'Profit'])}. "
    f"{best_region} is the strongest region by revenue. These rankings update with your filters.",
    icon=":material/lightbulb:",
)

overview, drivers, records = st.tabs(["Overview", "Performance drivers", "Transaction detail"])

with overview:
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader(f"Monthly {measure.lower()} trend")
            metric_col = METRICS[measure]
            monthly = filtered.groupby("date", as_index=False, observed=True)[metric_col].sum()
            monthly["date"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
            monthly = monthly.groupby("date", as_index=False)[metric_col].sum()
            line = alt.Chart(monthly).mark_area(
                line={"color": "#0F766E", "strokeWidth": 2.5},
                color=alt.Gradient(
                    gradient="linear", stops=[
                        {"offset": 0, "color": "#0F766E"},
                        {"offset": 1, "color": "#F7F9FC"},
                    ], x1=1, x2=1, y1=0, y2=1,
                ), opacity=0.18,
            ).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y(f"{metric_col}:Q", title=measure, axis=alt.Axis(format="$,.2s" if measure in ["Revenue", "Profit", "Ad spend"] else ",.0f")),
                tooltip=[alt.Tooltip("date:T", title="Month"), alt.Tooltip(f"{metric_col}:Q", title=measure, format="$,.2f" if measure in ["Revenue", "Profit", "Ad spend"] else ",.0f")],
            ).properties(height=300)
            st.altair_chart(chart_theme(line), width="stretch", alt=f"Monthly {measure.lower()} over the selected period")
    with right:
        with st.container(border=True):
            st.subheader("Revenue by category")
            cat_plot = cat_totals.reset_index().sort_values("Revenue")
            bar = alt.Chart(cat_plot).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X("Revenue:Q", title="Revenue", axis=alt.Axis(format="$,.2s")),
                y=alt.Y("Category:N", sort="-x", title=None),
                color=alt.Color("Category:N", scale=alt.Scale(range=COLORS), legend=None),
                tooltip=["Category", alt.Tooltip("Revenue:Q", format="$,.2f"), alt.Tooltip("Profit:Q", format="$,.2f"), "Transactions:Q"],
            ).properties(height=300)
            st.altair_chart(chart_theme(bar), width="stretch", alt="Revenue by category, ordered from highest to lowest")

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Category economics")
            cat_econ = cat_totals.reset_index().melt(
                id_vars="Category", value_vars=["Revenue", "Profit"],
                var_name="Measure", value_name="Amount",
            )
            econ = alt.Chart(cat_econ).mark_bar().encode(
                x=alt.X("Category:N", title=None, axis=alt.Axis(labelAngle=-28)),
                y=alt.Y("Amount:Q", title="Total", axis=alt.Axis(format="$,.2s")),
                color=alt.Color("Measure:N", scale=alt.Scale(domain=["Revenue", "Profit"], range=["#0F766E", "#93C5FD"]), title=None),
                xOffset="Measure:N",
                tooltip=["Category", "Measure", alt.Tooltip("Amount:Q", format="$,.2f")],
            ).properties(height=270)
            st.altair_chart(chart_theme(econ), width="stretch", alt="Revenue and profit totals by product category")
    with right:
        with st.container(border=True):
            st.subheader("Regional mix")
            region_plot = filtered.groupby("Region", as_index=False).agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"), Transactions=("Revenue", "size"))
            reg = alt.Chart(region_plot).mark_arc(innerRadius=72, outerRadius=112, stroke="#FFFFFF", strokeWidth=2).encode(
                theta=alt.Theta("Revenue:Q", stack=True),
                color=alt.Color("Region:N", scale=alt.Scale(range=COLORS), title="Region"),
                tooltip=["Region", alt.Tooltip("Revenue:Q", format="$,.2f"), "Transactions:Q"],
            ).properties(height=270)
            st.altair_chart(chart_theme(reg), width="stretch", alt="Share of revenue by region")

with drivers:
    st.subheader("What moves with sales?")
    st.caption("Use the filters to compare product mix, spend, and customer response within a consistent slice of the data.")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Units sold and revenue**")
            points = filtered.copy()
            scatter = alt.Chart(points).mark_circle(opacity=0.48, stroke="white", strokeWidth=0.5).encode(
                x=alt.X("Units_Sold:Q", title="Units sold"),
                y=alt.Y("Revenue:Q", title="Revenue", axis=alt.Axis(format="$,.2s")),
                color=alt.Color("Category:N", scale=alt.Scale(range=COLORS), title="Category"),
                size=alt.Size("Profit:Q", title="Profit", scale=alt.Scale(range=[25, 420]), legend=None),
                tooltip=["Category", "Region", "Year", "Month", "Units_Sold", alt.Tooltip("Revenue:Q", format="$,.2f"), alt.Tooltip("Profit:Q", format="$,.2f")],
            ).properties(height=320).interactive()
            st.altair_chart(chart_theme(scatter), width="stretch", alt="Each point is a transaction; revenue generally rises with units sold")
            corr = filtered["Units_Sold"].corr(filtered["Revenue"])
            st.caption(f"Within this selection, units sold and revenue have a Pearson correlation of **{corr:.2f}**. Correlation describes association, not cause.")
    with right:
        with st.container(border=True):
            st.markdown("**Ad spend and revenue**")
            ad = alt.Chart(filtered).mark_circle(opacity=0.5, stroke="white", strokeWidth=0.5).encode(
                x=alt.X("Ad_Spend:Q", title="Ad spend", axis=alt.Axis(format="$,.2s")),
                y=alt.Y("Revenue:Q", title="Revenue", axis=alt.Axis(format="$,.2s")),
                color=alt.Color("Region:N", scale=alt.Scale(range=COLORS), title="Region"),
                tooltip=["Category", "Region", "Year", alt.Tooltip("Ad_Spend:Q", format="$,.2f"), alt.Tooltip("Revenue:Q", format="$,.2f")],
            ).properties(height=320).interactive()
            st.altair_chart(chart_theme(ad), width="stretch", alt="Transaction revenue compared with advertising spend, colored by region")
            ad_corr = filtered["Ad_Spend"].corr(filtered["Revenue"])
            st.caption(f"Ad spend and revenue correlation in this slice: **{ad_corr:.2f}**. The dataset has no campaign or attribution identifiers, so this is descriptive only.")

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Rating distribution by category**")
            ratings = alt.Chart(filtered).mark_boxplot(size=34, extent="min-max").encode(
                x=alt.X("Customer_Rating:Q", title="Customer rating", scale=alt.Scale(domain=[0, 5])),
                y=alt.Y("Category:N", title=None),
                color=alt.Color("Category:N", scale=alt.Scale(range=COLORS), legend=None),
                tooltip=["Category"],
            ).properties(height=260)
            st.altair_chart(chart_theme(ratings), width="stretch", alt="Customer rating distributions across categories")
    with right:
        with st.container(border=True):
            st.markdown("**Seasonality by year**")
            heat_data = filtered.groupby(["Year", "Month"], observed=True, as_index=False)[METRICS[measure]].sum()
            heat = alt.Chart(heat_data).mark_rect(cornerRadius=3).encode(
                x=alt.X("Month:N", sort=MONTH_ORDER, title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Year:O", title=None),
                color=alt.Color(f"{METRICS[measure]}:Q", title=measure, scale=alt.Scale(scheme="tealblues")),
                tooltip=["Year", "Month", alt.Tooltip(f"{METRICS[measure]}:Q", title=measure, format="$,.2f" if measure in ["Revenue", "Profit", "Ad spend"] else ",.0f")],
            ).properties(height=260)
            st.altair_chart(chart_theme(heat), width="stretch", alt=f"Monthly {measure.lower()} intensity by year")

    with st.container(border=True):
        st.markdown("**Correlation across numeric fields**")
        correlation_cols = ["Units_Sold", "Revenue", "Profit", "Customer_Rating", "Ad_Spend"]
        corr_data = filtered[correlation_cols].corr().rename_axis("Field").reset_index().melt(id_vars="Field", var_name="Compared with", value_name="Correlation")
        corr_chart = alt.Chart(corr_data).mark_rect(cornerRadius=3).encode(
            x=alt.X("Compared with:N", title=None), y=alt.Y("Field:N", title=None),
            color=alt.Color("Correlation:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"), title="Correlation"),
            tooltip=["Field", "Compared with", alt.Tooltip("Correlation:Q", format=".2f")],
        ).properties(height=250)
        st.altair_chart(chart_theme(corr_chart), width="stretch", alt="Correlation matrix for units, revenue, profit, ratings, and ad spend")
    st.caption("As in the notebook, revenue and profit move closely together. Customer ratings and ad spend show much weaker linear relationships in this transaction sample.")

with records:
    st.subheader("Filtered transactions")
    st.caption("Sort or search the table, then download the filtered rows for follow-up analysis.")
    display = filtered.sort_values("date", ascending=False).copy()
    display["date"] = display["date"].dt.strftime("%Y-%m")
    display = display.rename(columns={
        "Year": "Year", "Month": "Month", "Category": "Category", "Region": "Region",
        "Units_Sold": "Units sold", "Revenue": "Revenue", "Profit": "Profit",
        "Customer_Rating": "Customer rating", "Ad_Spend": "Ad spend", "date": "Date",
    })
    st.dataframe(
        display[["Date", "Year", "Category", "Region", "Units sold", "Revenue", "Profit", "Customer rating", "Ad spend"]],
        hide_index=True,
        column_config={
            "Revenue": st.column_config.NumberColumn(format="$%.2f"),
            "Profit": st.column_config.NumberColumn(format="$%.2f"),
            "Ad spend": st.column_config.NumberColumn(format="$%.2f"),
            "Customer rating": st.column_config.NumberColumn(format="%.1f / 5"),
            "Units sold": st.column_config.NumberColumn(format="%d"),
        },
        height=520,
        alt="Transaction detail for the current filter selection",
    )
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name="filtered_ecommerce_transactions.csv", mime="text/csv", icon=":material/download:")

st.caption("Source: E-Commerce Sales Dataset referenced in EDA_1st_ipynb.ipynb · Transactions are observational; trend and correlation findings are descriptive.")
