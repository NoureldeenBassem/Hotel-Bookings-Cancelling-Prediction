"""
Hotel Booking Cancellation Prediction — Streamlit App
======================================================

DATA LEAKAGE SAFETY NOTE (read this before editing this file):
----------------------------------------------------------------
This app NEVER fits, trains, or refits anything. It only:
  1. Loads the single saved artifact `final_pipeline.joblib`, which already
     contains every preprocessing step (imputers, the custom FrequencyEncoder,
     OneHotEncoder, ColumnTransformer) AND the trained XGBoost model, fitted
     once on the training split inside the original notebook.
  2. Takes raw user inputs (the same raw columns the notebook started from).
  3. Reproduces the exact same deterministic `engineer_features()` function
     used in the notebook (pure arithmetic / mapping — no statistics learned
     from data, so it is safe to recompute at inference time).
  4. Calls `pipeline.predict_proba()` on the result. That's it.

Do NOT add `.fit(...)`, `.fit_transform(...)`, or re-create any encoder/scaler
in this file. Doing so would refit preprocessing on whatever single row the
user submits, which would silently corrupt every prediction.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------------------------------------------------------
# Custom transformer used inside the saved Pipeline.
#
# This class must be defined here, BEFORE joblib.load(), with the exact same
# name and implementation as in the training notebook. joblib/pickle stores a
# reference to "FrequencyEncoder" and re-attaches the *fitted* state
# (freq_maps_, columns_) that was learned once during training onto this
# class definition. The methods below are only ever called as `.transform()`
# during inference in this app — `.fit()` is never invoked here.
# ----------------------------------------------------------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Leakage-safe frequency encoder (definition must match training)."""

    def __init__(self):
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_maps_ = {col: X[col].value_counts(normalize=True).to_dict() for col in X.columns}
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        for col in X.columns:
            fmap = self.freq_maps_.get(col, {})
            X[col] = X[col].map(fmap).fillna(0.0)
        return X.values.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_])


# ----------------------------------------------------------------------------
# Paths / cached loaders
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
PIPELINE_PATH = APP_DIR / "final_pipeline.joblib"
CONFIG_PATH = APP_DIR / "deployment_config.json"

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

CATEGORY_OPTIONS = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "arrival_date_month": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "meal": ["BB", "FB", "HB", "SC", "Undefined"],
    "market_segment": [
        "Aviation", "Complementary", "Corporate", "Direct", "Groups",
        "Offline TA/TO", "Online TA", "Undefined",
    ],
    "distribution_channel": ["Corporate", "Direct", "GDS", "TA/TO", "Undefined"],
    "reserved_room_type": ["A", "B", "C", "D", "E", "F", "G", "H", "L"],
    "assigned_room_type": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Contract", "Group", "Transient", "Transient-Party"],
}

COUNTRY_OPTIONS = [
    "ABW", "AGO", "AIA", "ALB", "AND", "ARE", "ARG", "ARM", "ATA", "ATF", "AUS", "AUT", "AZE",
    "BEL", "BEN", "BFA", "BGD", "BGR", "BHR", "BHS", "BIH", "BLR", "BOL", "BRA", "BRB", "BWA",
    "CAF", "CHE", "CHL", "CHN", "CIV", "CMR", "CN", "COL", "COM", "CPV", "CRI", "CUB", "CYP",
    "CZE", "DEU", "DJI", "DMA", "DNK", "DOM", "DZA", "ECU", "EGY", "ESP", "EST", "ETH", "FIN",
    "FJI", "FRA", "FRO", "GAB", "GBR", "GEO", "GHA", "GIB", "GLP", "GNB", "GRC", "GTM", "HKG",
    "HND", "HRV", "HUN", "IDN", "IMN", "IND", "IRL", "IRN", "IRQ", "ISL", "ISR", "ITA", "JAM",
    "JEY", "JOR", "JPN", "KAZ", "KEN", "KHM", "KIR", "KNA", "KOR", "KWT", "LAO", "LBN", "LBY",
    "LCA", "LIE", "LKA", "LTU", "LUX", "LVA", "MAC", "MAR", "MCO", "MDG", "MDV", "MEX", "MKD",
    "MLI", "MLT", "MNE", "MOZ", "MRT", "MUS", "MWI", "MYS", "MYT", "NAM", "NCL", "NGA", "NIC",
    "NLD", "NOR", "NZL", "OMN", "PAK", "PAN", "PER", "PHL", "PLW", "POL", "PRI", "PRT", "PRY",
    "PYF", "QAT", "ROU", "RUS", "RWA", "SAU", "SDN", "SEN", "SGP", "SLV", "SMR", "SRB", "STP",
    "SUR", "SVK", "SVN", "SWE", "SYC", "SYR", "TGO", "THA", "TJK", "TMP", "TUN", "TUR", "TWN",
    "TZA", "UGA", "UKR", "UMI", "URY", "USA", "UZB", "Unknown", "VEN", "VGB", "VNM", "ZAF",
    "ZMB", "ZWE",
]


@st.cache_resource
def load_pipeline():
    """Load the single saved Pipeline object. Never fit anything here."""
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic feature engineering — identical to the training notebook.

    Every step here is pure arithmetic or a fixed dictionary lookup (no
    statistic is learned from the data), so it is safe to recompute at
    inference time on a single row without causing leakage. All *learned*
    preprocessing (imputation medians, frequency-encoding maps, one-hot
    categories) lives inside the saved Pipeline, not in this function.
    """
    frame = frame.copy()

    frame["total_nights"] = frame["stays_in_weekend_nights"] + frame["stays_in_week_nights"]
    frame["total_guests"] = frame["adults"] + frame["children"] + frame["babies"]
    frame["total_guests"] = frame["total_guests"].replace(0, 1)
    frame["is_family"] = (((frame["children"] > 0) | (frame["babies"] > 0)) & (frame["adults"] > 0)).astype(int)

    frame["room_changed"] = (frame["reserved_room_type"] != frame["assigned_room_type"]).astype(int)

    frame["total_previous_bookings"] = frame["previous_cancellations"] + frame["previous_bookings_not_canceled"]
    frame["prior_cancel_rate"] = frame["previous_cancellations"] / (frame["total_previous_bookings"] + 1)

    frame["has_agent"] = frame["agent"].notna().astype(int)
    frame["has_company"] = frame["company"].notna().astype(int)
    frame["agent"] = frame["agent"].fillna(-1).astype(int).astype(str)
    frame["company"] = frame["company"].fillna(-1).astype(int).astype(str)

    frame["month_num"] = frame["arrival_date_month"].map(MONTH_MAP)
    arrival_date = pd.to_datetime(
        dict(year=frame["arrival_date_year"], month=frame["month_num"], day=frame["arrival_date_day_of_month"]),
        errors="coerce",
    )
    frame["arrival_dow"] = arrival_date.dt.dayofweek
    frame["is_weekend_arrival"] = frame["arrival_dow"].isin([5, 6]).astype(int)

    frame["lead_time_log"] = np.log1p(frame["lead_time"])
    frame["adr_log"] = np.log1p(frame["adr"])
    frame["adr_per_person"] = frame["adr"] / frame["total_guests"]

    frame["is_direct_booking"] = (frame["distribution_channel"] == "Direct").astype(int)

    return frame


# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Hotel Booking Cancellation Predictor", page_icon="🏨", layout="centered")

pipeline = load_pipeline()
config = load_config()
FEATURE_COLUMNS = config["feature_columns"]
THRESHOLD = config["decision_threshold"]

st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("**Built by Noureldin Bassem** · Computer and AI Engineer")
st.caption(
    f"Model: **{config['model_name']}**  ·  "
    f"Test F1: **{config['test_f1']:.3f}**  ·  "
    f"Test ROC AUC: **{config['test_roc_auc']:.3f}**  ·  "
    f"Decision threshold: **{THRESHOLD:.2f}**"
)
st.write(
    "Fill in the booking details below. The trained pipeline (preprocessing + "
    "XGBoost model, saved exactly as produced by the training notebook) will "
    "estimate the probability that this booking gets **canceled**."
)

with st.form("booking_form"):
    st.subheader("Stay details")
    c1, c2, c3 = st.columns(3)
    with c1:
        hotel = st.selectbox("Hotel", CATEGORY_OPTIONS["hotel"])
    with c2:
        arrival_date_year = st.number_input("Arrival year", min_value=2000, max_value=2100, value=2017, step=1)
    with c3:
        arrival_date_month = st.selectbox("Arrival month", CATEGORY_OPTIONS["arrival_date_month"], index=6)

    c1, c2 = st.columns(2)
    with c1:
        arrival_date_week_number = st.number_input("Arrival week number", min_value=1, max_value=53, value=27, step=1)
    with c2:
        arrival_date_day_of_month = st.number_input("Arrival day of month", min_value=1, max_value=31, value=1, step=1)

    c1, c2 = st.columns(2)
    with c1:
        stays_in_weekend_nights = st.number_input("Weekend nights", min_value=0, value=2, step=1)
    with c2:
        stays_in_week_nights = st.number_input("Week nights", min_value=0, value=3, step=1)

    lead_time = st.number_input("Lead time (days between booking and arrival)", min_value=0, value=45, step=1)

    st.subheader("Guests")
    c1, c2, c3 = st.columns(3)
    with c1:
        adults = st.number_input("Adults", min_value=0, value=2, step=1)
    with c2:
        children = st.number_input("Children", min_value=0, value=0, step=1)
    with c3:
        babies = st.number_input("Babies", min_value=0, value=0, step=1)

    st.subheader("Booking channel & guest profile")
    c1, c2 = st.columns(2)
    with c1:
        meal = st.selectbox("Meal plan", CATEGORY_OPTIONS["meal"])
        market_segment = st.selectbox("Market segment", CATEGORY_OPTIONS["market_segment"])
        distribution_channel = st.selectbox("Distribution channel", CATEGORY_OPTIONS["distribution_channel"])
    with c2:
        country = st.selectbox("Country (ISO code)", COUNTRY_OPTIONS, index=COUNTRY_OPTIONS.index("PRT"))
        customer_type = st.selectbox("Customer type", CATEGORY_OPTIONS["customer_type"])
        is_repeated_guest = st.selectbox("Repeated guest?", ["No", "Yes"]) == "Yes"

    st.subheader("Rooms")
    c1, c2 = st.columns(2)
    with c1:
        reserved_room_type = st.selectbox("Reserved room type", CATEGORY_OPTIONS["reserved_room_type"])
    with c2:
        assigned_room_type = st.selectbox("Assigned room type", CATEGORY_OPTIONS["assigned_room_type"])
    booking_changes = st.number_input("Number of booking changes", min_value=0, value=0, step=1)

    st.subheader("Guest history")
    c1, c2 = st.columns(2)
    with c1:
        previous_cancellations = st.number_input("Previous cancellations", min_value=0, value=0, step=1)
    with c2:
        previous_bookings_not_canceled = st.number_input("Previous bookings not canceled", min_value=0, value=0, step=1)

    st.subheader("Agent / company")
    c1, c2 = st.columns(2)
    with c1:
        has_agent_input = st.checkbox("Booked through an agent", value=False)
        agent_id = st.number_input("Agent ID (used only if checked above)", min_value=0, value=1, step=1)
    with c2:
        has_company_input = st.checkbox("Booked through a company", value=False)
        company_id = st.number_input("Company ID (used only if checked above)", min_value=0, value=1, step=1)

    st.subheader("Payment & pricing")
    c1, c2 = st.columns(2)
    with c1:
        deposit_type = st.selectbox("Deposit type", CATEGORY_OPTIONS["deposit_type"])
        days_in_waiting_list = st.number_input("Days in waiting list", min_value=0, value=0, step=1)
    with c2:
        adr = st.number_input("Average daily rate (ADR)", min_value=0.0, value=100.0, step=1.0)
        required_car_parking_spaces = st.number_input("Required car parking spaces", min_value=0, value=0, step=1)

    total_of_special_requests = st.number_input("Total special requests", min_value=0, value=1, step=1)

    submitted = st.form_submit_button("Predict cancellation risk", use_container_width=True)

if submitted:
    raw_input = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": int(is_repeated_guest),
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": float(agent_id) if has_agent_input else np.nan,
        "company": float(company_id) if has_company_input else np.nan,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
    }

    input_df = pd.DataFrame([raw_input])
    engineered_df = engineer_features(input_df)

    # Reindex to the exact column set/order the pipeline was trained on.
    # No fitting happens here — this only selects/orders columns.
    model_input = engineered_df.reindex(columns=FEATURE_COLUMNS)

    # The saved Pipeline performs ALL preprocessing (imputation, frequency
    # encoding, one-hot encoding) internally via .transform(), then the
    # trained XGBoost model scores the result. Nothing is fit here.
    probability = float(pipeline.predict_proba(model_input)[:, 1][0])
    prediction = int(probability >= THRESHOLD)

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Likely to be **CANCELED** — predicted probability: **{probability:.1%}**")
    else:
        st.success(f"✅ Likely to be **HONORED** — predicted cancellation probability: **{probability:.1%}**")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Decision threshold in use: {THRESHOLD:.2f} (probability ≥ threshold ⇒ predicted cancellation)")

    with st.expander("Show engineered features sent to the model"):
        st.dataframe(model_input.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "This app loads a single pre-trained scikit-learn Pipeline "
    "(`final_pipeline.joblib`) and only calls `predict_proba()` on it — "
    "no preprocessing step is ever fit inside this app."
)
