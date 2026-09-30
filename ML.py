# %%
# # Freight Rate Prediction – ML Engineer assessment
# 


# %%
import os, sys, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DATA_DIR")
SEARCH = [Path(DATA_DIR)] if DATA_DIR else [Path("data"), Path(".")]

def find(*names):
    for d in SEARCH:
        for n in names:
            if (d / n).is_file():
                return d / n
    raise FileNotFoundError(f"Could not find any of {names} in {[str(d) for d in SEARCH]}")

train = pd.read_csv(find("train_test.csv", "train-test.csv"), parse_dates=["date"])
valid = pd.read_csv(find("validation.csv"), parse_dates=["date"])
template = pd.read_csv(find("validation_predictions_template.csv", "validation-predictions-template.csv"))
dec_in = pd.read_csv(find("december_chart_inputs.csv", "december-chart-inputs.csv"))
SCORER = find("score.py")
print(train.shape, valid.shape, template.shape, dec_in.shape)

# %%
# ## 1. Data exploration and data-quality checks
# Key facts this section verifies (all printed below):
# * **Train covers 2025-01-01 → 2025-10-31; validation covers 2025-11-01 → 2025-12-31.** The final task is forecasting the *future*, so a random split would leak and be over-optimistic → I validate with a **time-based split**.
# * **12 % of validation loads use cities never seen in training** (e.g. Chicago, Charlotte, Laredo). City-ID encoding cannot work for them, so the model uses coordinates/distance instead of city IDs.
# * Data issues: negative weights (sign errors), weight capped at 47,500, missing weight / market_index, and corrupted target values (rate per mile ~3-6x or ~0.3x normal).

# %%
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))

print("Date range  train:", train.date.min().date(), "->", train.date.max().date())
print("Date range  valid:", valid.date.min().date(), "->", valid.date.max().date())
print("Missing (train):", train.isna().sum()[lambda s: s > 0].to_dict())
print("Missing (valid):", valid.isna().sum()[lambda s: s > 0].to_dict())
print("Duplicate load_id:", train.load_id.duplicated().sum(), "| exact duplicate rows:", train.drop(columns="load_id").duplicated().sum())
unseen = (set(valid.pickup) | set(valid.delivery)) - (set(train.pickup) | set(train.delivery))
print("Cities only in validation:", sorted(unseen))
print("Validation rows touching an unseen city: %.1f%%" % (100 * (valid.pickup.isin(unseen) | valid.delivery.isin(unseen)).mean()))
print("Negative weights train/valid:", (train.weight < 0).sum(), (valid.weight < 0).sum(), "| weight == 47,500 (cap):", (train.weight == 47500).sum())
for name, d in [("train", train), ("valid", valid)]:
    ratio = d.distance / haversine(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    print(f"Road/straight-line distance ratio ({name}): median {ratio.median():.2f}, long hauls (>150 mi) above 2.0: {((ratio > 2) & (d.distance > 150)).sum()}")
rpm = train.posted_rate / train.distance
print("Rate/mile median %.2f | >4: %d rows | <0.9: %d rows (corrupted-looking targets)" % (rpm.median(), (rpm > 4).sum(), (rpm < 0.9).sum()))
print("Median rate/mile by equipment:", rpm.groupby(train.equipment).median().round(2).to_dict())
print("Median rate/mile by distance band:", rpm.groupby(pd.cut(train.distance, [0, 150, 300, 500, 1000, 2000, 4000])).median().round(2).to_dict())
print("Corr(rate, distance) = %.2f" % train[["posted_rate", "distance"]].corr().iloc[0, 1])

# %%
## 2. Cleaning and feature engineering
# Modelling choice: `rate ≈ distance × rate_per_mile`, so the model predicts **log(rate per mile)** and multiplies back by distance.
# This removes the dominant distance effect, stabilises the skewed target and makes errors proportional.

# * `weight`: absolute value (sign errors), missing → median + missing flag
# * `market_index`: missing → median index of all loads on the same date (it is a date-level signal), + missing flag
# * `distance` was checked against straight-line (haversine) distance: median ratio is 1.18 in train and validation with no long-haul outliers, so it is used as given
# * `equipment`: fixed category order (Dry Van, Flatbed, Reefer) 
# * Route coordinates + haversine distance instead of city IDs, so unseen cities still get sensible predictions
# * No calendar-trend feature such as month / day-of-year: it can't extrapolate to Nov-Dec and made time-based validation unstable

# %%
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
FEATURES = ["ldist", "dist_fixed", "hv", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "eq", "w", "w_na", "mi", "mi_na", "quote_signal", "dow"]
CAT_IDX = [FEATURES.index("eq")]

def make_features(df, mi_by_date, weight_fill):
    d = df.copy()
    d["hv"] = haversine(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    d["dist_fixed"] = d.distance
    d["ldist"] = np.log(d.dist_fixed)
    w = d.weight.abs().clip(upper=47500)
    d["w_na"] = w.isna().astype(int)
    d["w"] = w.fillna(weight_fill)
    d["mi_na"] = d.market_index.isna().astype(int)
    d["mi"] = d.market_index.fillna(d.date.map(mi_by_date)).fillna(mi_by_date.median())
    d["eq"] = d.equipment.map({e: i for i, e in enumerate(EQUIPMENT)})
    d["dow"] = d.date.dt.dayofweek
    return d

PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
              l2_regularization=1.0, random_state=0, categorical_features=CAT_IDX)
HALF_LIFE_DAYS = 30  

def fit_predict(train_df, test_df):
    """Train on train_df (with robust label cleaning + recency weights) and predict rates for test_df."""
    mi = pd.concat([train_df, test_df]).groupby("date").market_index.median()
    wfill = train_df.weight.abs().median()
    a = make_features(train_df, mi, wfill)
    b = make_features(test_df, mi, wfill)
    y = np.log(a.posted_rate / a.dist_fixed)
    weights = 0.5 ** ((a.date.max() - a.date).dt.days.values / HALF_LIFE_DAYS)
    # pass 1: robust (absolute-error) model to spot corrupted targets
    m0 = HistGradientBoostingRegressor(loss="absolute_error", **PARAMS).fit(a[FEATURES], y, sample_weight=weights)
    r = y - m0.predict(a[FEATURES])
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = (np.abs(r) < 4 * sigma).values
    # pass 2: final model on cleaned labels
    m = HistGradientBoostingRegressor(**PARAMS).fit(a.loc[keep, FEATURES], y[keep], sample_weight=weights[keep])
    return np.exp(m.predict(b[FEATURES])) * b.dist_fixed.values, keep.mean()

def metrics(y, p):
    e = p - y
    return {"MAE": np.abs(e).mean(), "MAPE_%": 100 * np.mean(np.abs(e) / y), "medAPE_%": 100 * np.median(np.abs(e) / y),
            "RMSE": np.sqrt((e ** 2).mean()), "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}

# %%
# ## 3. Validation strategy (time-based, no random split)
# Validation data is *future* data (Nov-Dec), so I mimic that with **expanding-window folds**: train on everything before the fold, test on the next two months
# (Jul-Aug and Sep-Oct). Metrics are reported on all rows (raw, includes corrupted labels which nobody can predict) and on "inlier" rows.
# A naive baseline (global median rate-per-mile × distance) shows the value of the model.

# %%
train = train.sort_values("date").reset_index(drop=True)
FOLDS = [("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]
rows = []
for start, end in FOLDS:
    A = train[train.date < start]
    B = train[(train.date >= start) & (train.date < end)]
    y = B.posted_rate.values
    pred, kept = fit_predict(A, B)
    base = B.distance.values * (A.posted_rate / A.distance).median()
    rpm_b = (B.posted_rate / B.distance).values
    inl = (rpm_b < 4) & (rpm_b > 0.9)
    window = f"{start[:7]}..{end[:7]}"
    rows.append({"test_window": window, "setup": "model (all rows)", **metrics(y, pred)})
    rows.append({"test_window": window, "setup": "model (inlier rows)", **metrics(y[inl], pred[inl])})
    rows.append({"test_window": window, "setup": "baseline (all rows)", **metrics(y, base)})
    print(f"fold {start[:7]}: training rows kept after label cleaning = {kept:.1%}")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %%
# ## 4. Final model and validation-set predictions
# Train on **all** of `train_test.csv` with exactly the configuration validated above, then predict `validation.csv` in the template's order.

# %%
val_pred, kept = fit_predict(train, valid)
print(f"final model trained; label rows kept after cleaning: {kept:.1%}")

out = pd.DataFrame({"load_id": valid.load_id, "predicted_rate": val_pred})
out = template[["load_id"]].merge(out, on="load_id", how="left")     
assert len(out) == 12000 and out.load_id.is_unique
assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
assert list(out.columns) == ["load_id", "predicted_rate"]
out.to_csv("validation_predictions.csv", index=False)
print(out.predicted_rate.describe().round(1).to_dict())

# %%
# ## 5. December chart predictions
# The December file only gives route, distance, equipment, weight and date. The model also needs `market_index` and `quote_signal`, so they are taken from the
# validation loads on the *same calendar date* (daily median) rather than from a global training mean – the earlier version overwrote them with a constant.
# Coordinates come from the city lookup built from the data. The provided input file is **not** modified; results go to `december_predictions.csv`.

# %%
cities = pd.concat([
    train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
]).drop_duplicates("city").set_index("city")

dec = dec_in.copy()
dec["date"] = pd.to_datetime(dec["date"])
daily = valid.groupby("date").agg(market_index=("market_index", "median"), quote_signal=("quote_signal", "median"))
dec = dec.join(daily, on="date")
assert dec[["market_index", "quote_signal"]].notna().all().all(), "no market data for some December dates"
dec["pickup_lat"] = dec.pickup.map(cities.lat); dec["pickup_lon"] = dec.pickup.map(cities.lon)
dec["delivery_lat"] = dec.delivery.map(cities.lat); dec["delivery_lon"] = dec.delivery.map(cities.lon)

dec_pred, _ = fit_predict(train, dec.drop(columns="predicted_rate"))
dec_out = dec_in.copy()
dec_out["predicted_rate"] = dec_pred
dec_out.to_csv("december_predictions.csv", index=False)
print(dec_out.predicted_rate.describe().round(1).to_dict())

# %%
# ## 6. Run the provided scorer

# %%
res = subprocess.run([sys.executable, str(SCORER), "--predictions", "validation_predictions.csv",
                      "--december-predictions", "december_predictions.csv"], capture_output=True, text=True)
print(res.stdout, res.stderr)
