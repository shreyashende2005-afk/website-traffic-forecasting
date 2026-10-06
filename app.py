import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Website Traffic Forecasting",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f8fc;
}

.block-container {
    padding-top: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #173f7a, #2563a8);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.06);
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>📈 Website Traffic Forecasting</h1>

<p>
Time-Series + Machine Learning Dashboard for
Daily Website Traffic Prediction
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("daily-website-visitors.csv")

    # Convert numeric columns
    numeric_columns = [
        "Page.Loads",
        "Unique.Visits",
        "First.Time.Visits",
        "Returning.Visits"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False),
            errors="coerce"
        )

    # Convert date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Sort data
    df = df.sort_values("Date")

    # Remove duplicate dates
    df = df.drop_duplicates("Date")

    # Create daily time series
    traffic = (
        df.set_index("Date")["Page.Loads"]
        .asfreq("D")
    )

    # Fill missing values
    traffic = traffic.interpolate(
        limit_direction="both"
    )

    return df, traffic


df, traffic = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Dashboard Controls")

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=7,
    max_value=60,
    value=30
)

show_data = st.sidebar.checkbox(
    "Show Dataset"
)


# ============================================================
# KPI SECTION
# ============================================================

total_days = len(traffic)

average_traffic = traffic.mean()

maximum_traffic = traffic.max()

latest_traffic = traffic.iloc[-1]


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📊 Total Days",
        f"{total_days:,}"
    )


with col2:

    st.metric(
        "👁️ Average Traffic",
        f"{average_traffic:,.0f}"
    )


with col3:

    st.metric(
        "🚀 Maximum Traffic",
        f"{maximum_traffic:,.0f}"
    )


with col4:

    st.metric(
        "📅 Latest Traffic",
        f"{latest_traffic:,.0f}"
    )


st.caption(
    f"Data Period: "
    f"{traffic.index.min().strftime('%d %b %Y')} "
    f"to "
    f"{traffic.index.max().strftime('%d %b %Y')}"
)


# ============================================================
# SHOW DATA
# ============================================================

if show_data:

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )


# ============================================================
# TRAFFIC TREND
# ============================================================

st.subheader("📈 Daily Website Traffic")

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    traffic.index,
    traffic.values
)

ax.set_title(
    "Daily Website Page Loads"
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "Page Loads"
)

ax.grid(
    alpha=0.25
)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# DAY OF WEEK ANALYSIS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "📅 Average Traffic by Day"
    )

    weekday = (
        traffic
        .groupby(traffic.index.dayofweek)
        .mean()
    )

    weekday.index = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    weekday.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel(
        "Average Page Loads"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    st.pyplot(fig)

    plt.close(fig)


with col2:

    st.subheader(
        "📊 Monthly Average Traffic"
    )

    monthly = (
        traffic
        .resample("ME")
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        monthly.index,
        monthly.values
    )

    ax.set_ylabel(
        "Average Page Loads"
    )

    ax.grid(
        alpha=0.25
    )

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(series):

    data = pd.DataFrame(
        index=series.index
    )

    data["traffic"] = series

    # Lag features
    for lag in [
        1,
        2,
        3,
        7,
        14,
        21,
        28
    ]:

        data[f"lag_{lag}"] = (
            series.shift(lag)
        )

    # Rolling statistics
    for window in [
        7,
        14,
        28
    ]:

        data[f"rolling_mean_{window}"] = (
            series
            .shift(1)
            .rolling(window)
            .mean()
        )

        data[f"rolling_std_{window}"] = (
            series
            .shift(1)
            .rolling(window)
            .std()
        )

    # Calendar features
    data["day_of_week"] = (
        data.index.dayofweek
    )

    data["day_of_month"] = (
        data.index.day
    )

    data["month"] = (
        data.index.month
    )

    data["week_of_year"] = (
        data.index.isocalendar()
        .week
        .astype(int)
    )

    # Trend
    data["trend"] = (
        data.index - data.index.min()
    ).days

    return data.dropna()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    actual,
    predicted
):

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mape = np.mean(
        np.abs(
            (actual - predicted)
            / np.maximum(
                np.abs(actual),
                1e-8
            )
        )
    ) * 100

    return rmse, mae, mape


# ============================================================
# CREATE FEATURES
# ============================================================

features = create_features(
    traffic
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(traffic) * 0.80
)

test_start = traffic.index[
    split_index
]


train_data = features[
    features.index < test_start
]

test_data = features[
    features.index >= test_start
]


X_train = train_data.drop(
    columns=["traffic"]
)

y_train = train_data["traffic"]


X_test = test_data.drop(
    columns=["traffic"]
)

y_test = test_data["traffic"]


# ============================================================
# MODEL 1 — WEEKLY NAIVE
# ============================================================

weekly_naive = (
    traffic
    .shift(7)
    .loc[y_test.index]
)


# ============================================================
# MODEL 2 — HOLT-WINTERS
# ============================================================

train_series = traffic[
    traffic.index < test_start
]


holt_winters = ExponentialSmoothing(
    train_series,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
    initialization_method="estimated"
).fit()


holt_predictions = (
    holt_winters
    .forecast(len(y_test))
)


# ============================================================
# MODEL 3 — RANDOM FOREST
# ============================================================

random_forest = RandomForestRegressor(

    n_estimators=150,

    max_depth=12,

    min_samples_leaf=2,

    max_features=0.8,

    random_state=42,

    n_jobs=-1

)


random_forest.fit(
    X_train,
    y_train
)


rf_predictions = (
    random_forest
    .predict(X_test)
)


# ============================================================
# MODEL 4 — EXTRA TREES
# ============================================================

extra_trees = ExtraTreesRegressor(

    n_estimators=250,

    max_depth=16,

    min_samples_leaf=2,

    max_features=0.9,

    random_state=42,

    n_jobs=-1

)


extra_trees.fit(
    X_train,
    y_train
)


extra_predictions = (
    extra_trees
    .predict(X_test)
)


# ============================================================
# MODEL 5 — HIST GRADIENT BOOSTING
# ============================================================

hist_gradient = (
    HistGradientBoostingRegressor(

        max_iter=300,

        learning_rate=0.05,

        max_leaf_nodes=31,

        l2_regularization=1,

        random_state=42

    )
)


hist_gradient.fit(
    X_train,
    y_train
)


hist_predictions = (
    hist_gradient
    .predict(X_test)
)


# ============================================================
# MODEL COMPARISON
# ============================================================

predictions = {

    "Weekly Naive":
        weekly_naive,

    "Holt-Winters":
        holt_predictions,

    "Random Forest":
        rf_predictions,

    "Extra Trees":
        extra_predictions,

    "HistGradientBoosting":
        hist_predictions

}


results = []


for model_name, prediction in predictions.items():

    rmse, mae, mape = evaluate_model(
        y_test,
        prediction
    )

    results.append({

        "Model": model_name,

        "RMSE": rmse,

        "MAE": mae,

        "MAPE (%)": mape

    })


comparison = pd.DataFrame(
    results
).sort_values(
    "RMSE"
)


# ============================================================
# DISPLAY MODEL RESULTS
# ============================================================

st.subheader(
    "🤖 Model Performance Comparison"
)


st.dataframe(
    comparison.style.format({

        "RMSE": "{:,.2f}",

        "MAE": "{:,.2f}",

        "MAPE (%)": "{:.2f}%"

    }),
    use_container_width=True
)


best_model = comparison.iloc[0]["Model"]

best_rmse = comparison.iloc[0]["RMSE"]


st.success(
    f"🏆 Best Model: {best_model} | "
    f"RMSE: {best_rmse:,.2f}"
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.subheader(
    "📊 Actual vs Predicted Traffic"
)


best_prediction = predictions[
    best_model
]


fig, ax = plt.subplots(
    figsize=(14, 5)
)


ax.plot(
    y_test.index,
    y_test.values,
    label="Actual"
)


ax.plot(
    y_test.index,
    best_prediction,
    label=f"Predicted - {best_model}"
)


ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Page Loads"
)

ax.legend()

ax.grid(
    alpha=0.25
)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔍 Feature Importance — Extra Trees"
)


importance = pd.Series(

    extra_trees.feature_importances_,

    index=X_train.columns

).sort_values(
    ascending=False
).head(15)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


importance.sort_values().plot(
    kind="barh",
    ax=ax
)


ax.set_xlabel(
    "Importance"
)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# FUTURE FORECAST
# ============================================================

st.subheader(
    "🔮 Future Website Traffic Forecast"
)


def generate_future_forecast(
    series,
    model,
    days
):

    history = series.copy()

    future_dates = pd.date_range(

        start=
        history.index[-1]
        + pd.Timedelta(days=1),

        periods=days,

        freq="D"

    )

    forecast_values = []


    for date in future_dates:

        row = {

            "lag_1":
                history.iloc[-1],

            "lag_2":
                history.iloc[-2],

            "lag_3":
                history.iloc[-3],

            "lag_7":
                history.iloc[-7],

            "lag_14":
                history.iloc[-14],

            "lag_21":
                history.iloc[-21],

            "lag_28":
                history.iloc[-28],

            "rolling_mean_7":
                history.iloc[-7:].mean(),

            "rolling_std_7":
                history.iloc[-7:].std(),

            "rolling_mean_14":
                history.iloc[-14:].mean(),

            "rolling_std_14":
                history.iloc[-14:].std(),

            "rolling_mean_28":
                history.iloc[-28:].mean(),

            "rolling_std_28":
                history.iloc[-28:].std(),

            "day_of_week":
                date.dayofweek,

            "day_of_month":
                date.day,

            "month":
                date.month,

            "week_of_year":
                int(
                    date.isocalendar().week
                ),

            "trend":
                (
                    date
                    - history.index.min()
                ).days

        }


        input_data = pd.DataFrame(
            [row],
            index=[date]
        )


        prediction = model.predict(
            input_data
        )[0]


        prediction = max(
            0,
            prediction
        )


        history.loc[date] = prediction

        forecast_values.append(
            prediction
        )


    forecast = pd.DataFrame({

        "Date": future_dates,

        "Forecasted Page Loads":
            forecast_values

    })


    return forecast


future_forecast = generate_future_forecast(

    traffic,

    extra_trees,

    forecast_days

)


# ============================================================
# FORECAST GRAPH
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 5)
)


recent_days = min(
    90,
    len(traffic)
)


ax.plot(

    traffic.index[-recent_days:],

    traffic.values[-recent_days:],

    label="Historical Traffic"

)


ax.plot(

    future_forecast["Date"],

    future_forecast[
        "Forecasted Page Loads"
    ],

    linestyle="--",

    label="Future Forecast"

)


ax.set_title(
    f"Next {forecast_days} Days Website Traffic Forecast"
)


ax.set_xlabel(
    "Date"
)


ax.set_ylabel(
    "Page Loads"
)


ax.legend()


ax.grid(
    alpha=0.25
)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# FORECAST TABLE
# ============================================================

st.dataframe(

    future_forecast.style.format({

        "Forecasted Page Loads":
            "{:,.0f}"

    }),

    use_container_width=True

)


# ============================================================
# DOWNLOAD FORECAST
# ============================================================

csv_data = (
    future_forecast
    .to_csv(index=False)
)


st.download_button(

    label="⬇️ Download Forecast CSV",

    data=csv_data,

    file_name=
    "website_traffic_forecast.csv",

    mime="text/csv"

)


# ============================================================
# PROJECT METHODOLOGY
# ============================================================

st.subheader(
    "🧠 Project Methodology"
)


st.markdown("""

### Data Preprocessing

- Date formatting and chronological sorting
- Daily time-series structure
- Numeric conversion of traffic columns
- Missing-value handling

### Time-Series Models

- Weekly Seasonal Naive
- Holt-Winters Exponential Smoothing

### Machine Learning Models

- Random Forest Regressor
- Extra Trees Regressor
- HistGradientBoosting Regressor

### Feature Engineering

- Lag 1, 2, 3, 7, 14, 21 and 28 days
- Rolling mean and standard deviation
- Day of week
- Day of month
- Month
- Week of year
- Trend

### Performance Evaluation

- RMSE
- MAE
- MAPE

An 80/20 chronological split is used instead of random splitting
to avoid future-data leakage.

""")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Website Traffic Forecasting | "
    "Python • Streamlit • Scikit-learn • Statsmodels"
)