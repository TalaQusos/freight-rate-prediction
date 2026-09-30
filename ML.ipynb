{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "52746d4c",
   "metadata": {},
   "source": [
    "# Freight Rate Prediction \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0eb30567",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(48000, 14) (12000, 13) (12000, 2) (31, 7)\n"
     ]
    }
   ],
   "source": [
    "import os, sys, subprocess, warnings\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import HistGradientBoostingRegressor\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "DATA_DIR = os.environ.get(\"DATA_DIR\")\n",
    "SEARCH = [Path(DATA_DIR)] if DATA_DIR else [Path(\"data\"), Path(\".\")]\n",
    "\n",
    "def find(*names):\n",
    "    for d in SEARCH:\n",
    "        for n in names:\n",
    "            if (d / n).is_file():\n",
    "                return d / n\n",
    "    raise FileNotFoundError(f\"Could not find any of {names} in {[str(d) for d in SEARCH]}\")\n",
    "\n",
    "train = pd.read_csv(find(\"train_test.csv\", \"train-test.csv\"), parse_dates=[\"date\"])\n",
    "valid = pd.read_csv(find(\"validation.csv\"), parse_dates=[\"date\"])\n",
    "template = pd.read_csv(find(\"validation_predictions_template.csv\", \"validation-predictions-template.csv\"))\n",
    "dec_in = pd.read_csv(find(\"december_chart_inputs.csv\", \"december-chart-inputs.csv\"))\n",
    "SCORER = find(\"score.py\")\n",
    "print(train.shape, valid.shape, template.shape, dec_in.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a53b777",
   "metadata": {},
   "source": [
    "## 1. Data exploration and data-quality checks\n",
    "Key facts this section verifies (all printed below):\n",
    "* **Train covers 2025-01-01 → 2025-10-31; validation covers 2025-11-01 → 2025-12-31.** The final task is forecasting the *future*, so a random split would leak and be over-optimistic → I validate with a **time-based split**.\n",
    "* **12 % of validation loads use cities never seen in training** (e.g. Chicago, Charlotte, Laredo). City-ID encoding cannot work for them, so the model uses coordinates/distance instead of city IDs.\n",
    "* Data issues: negative weights (sign errors), weight capped at 47,500, missing weight / market_index, and corrupted target values (rate per mile ~3-6x or ~0.3x normal)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "43c4436e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Date range  train: 2025-01-01 -> 2025-10-31\n",
      "Date range  valid: 2025-11-01 -> 2025-12-31\n",
      "Missing (train): {'weight': 300, 'market_index': 374}\n",
      "Missing (valid): {'weight': 165, 'market_index': 249}\n",
      "Duplicate load_id: 0 | exact duplicate rows: 0\n",
      "Cities only in validation: ['Allentown', 'Charlotte', 'Chicago', 'Jackson', 'Knoxville', 'Laredo', 'Norfolk', 'San Diego']\n",
      "Validation rows touching an unseen city: 12.1%\n",
      "Negative weights train/valid: 292 145 | weight == 47,500 (cap): 1191\n",
      "Road/straight-line distance ratio (train): median 1.18, long hauls (>150 mi) above 2.0: 0\n",
      "Road/straight-line distance ratio (valid): median 1.18, long hauls (>150 mi) above 2.0: 0\n",
      "Rate/mile median 2.15 | >4: 340 rows | <0.9: 307 rows (corrupted-looking targets)\n",
      "Median rate/mile by equipment: {'Dry Van': 2.05, 'Flatbed': 2.22, 'Reefer': 2.31}\n",
      "Median rate/mile by distance band: {Interval(0, 150, closed='right'): 2.87, Interval(150, 300, closed='right'): 2.63, Interval(300, 500, closed='right'): 2.4, Interval(500, 1000, closed='right'): 2.2, Interval(1000, 2000, closed='right'): 2.05, Interval(2000, 4000, closed='right'): 1.91}\n",
      "Corr(rate, distance) = 0.91\n"
     ]
    }
   ],
   "source": [
    "def haversine(lat1, lon1, lat2, lon2):\n",
    "    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])\n",
    "    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2\n",
    "    return 3958.8 * 2 * np.arcsin(np.sqrt(a))\n",
    "\n",
    "print(\"Date range  train:\", train.date.min().date(), \"->\", train.date.max().date())\n",
    "print(\"Date range  valid:\", valid.date.min().date(), \"->\", valid.date.max().date())\n",
    "print(\"Missing (train):\", train.isna().sum()[lambda s: s > 0].to_dict())\n",
    "print(\"Missing (valid):\", valid.isna().sum()[lambda s: s > 0].to_dict())\n",
    "print(\"Duplicate load_id:\", train.load_id.duplicated().sum(), \"| exact duplicate rows:\", train.drop(columns=\"load_id\").duplicated().sum())\n",
    "unseen = (set(valid.pickup) | set(valid.delivery)) - (set(train.pickup) | set(train.delivery))\n",
    "print(\"Cities only in validation:\", sorted(unseen))\n",
    "print(\"Validation rows touching an unseen city: %.1f%%\" % (100 * (valid.pickup.isin(unseen) | valid.delivery.isin(unseen)).mean()))\n",
    "print(\"Negative weights train/valid:\", (train.weight < 0).sum(), (valid.weight < 0).sum(), \"| weight == 47,500 (cap):\", (train.weight == 47500).sum())\n",
    "for name, d in [(\"train\", train), (\"valid\", valid)]:\n",
    "    ratio = d.distance / haversine(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)\n",
    "    print(f\"Road/straight-line distance ratio ({name}): median {ratio.median():.2f}, long hauls (>150 mi) above 2.0: {((ratio > 2) & (d.distance > 150)).sum()}\")\n",
    "rpm = train.posted_rate / train.distance\n",
    "print(\"Rate/mile median %.2f | >4: %d rows | <0.9: %d rows (corrupted-looking targets)\" % (rpm.median(), (rpm > 4).sum(), (rpm < 0.9).sum()))\n",
    "print(\"Median rate/mile by equipment:\", rpm.groupby(train.equipment).median().round(2).to_dict())\n",
    "print(\"Median rate/mile by distance band:\", rpm.groupby(pd.cut(train.distance, [0, 150, 300, 500, 1000, 2000, 4000])).median().round(2).to_dict())\n",
    "print(\"Corr(rate, distance) = %.2f\" % train[[\"posted_rate\", \"distance\"]].corr().iloc[0, 1])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d08eb54",
   "metadata": {},
   "source": [
    "## 2. Cleaning and feature engineering\n",
    "Modelling choice: `rate ≈ distance × rate_per_mile`, so the model predicts **log(rate per mile)** and multiplies back by distance.\n",
    "This removes the dominant distance effect, stabilises the skewed target and makes errors proportional.\n",
    "\n",
    "* `weight`: absolute value (sign errors), missing → median + missing flag\n",
    "* `market_index`: missing → median index of all loads on the same date (it is a date-level signal), + missing flag\n",
    "* `distance` was checked against straight-line (haversine) distance: median ratio is 1.18 in train and validation with no long-haul outliers, so it is used as given\n",
    "* `equipment`: fixed category order (Dry Van, Flatbed, Reefer) \n",
    "* Route coordinates + haversine distance instead of city IDs, so unseen cities still get sensible predictions\n",
    "* No calendar-trend feature such as month / day-of-year: it can't extrapolate to Nov-Dec and made time-based validation unstable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "313fade4",
   "metadata": {},
   "outputs": [],
   "source": [
    "EQUIPMENT = [\"Dry Van\", \"Flatbed\", \"Reefer\"]\n",
    "FEATURES = [\"ldist\", \"dist_fixed\", \"hv\", \"pickup_lat\", \"pickup_lon\", \"delivery_lat\", \"delivery_lon\",\n",
    "            \"eq\", \"w\", \"w_na\", \"mi\", \"mi_na\", \"quote_signal\", \"dow\"]\n",
    "CAT_IDX = [FEATURES.index(\"eq\")]\n",
    "\n",
    "def make_features(df, mi_by_date, weight_fill):\n",
    "    d = df.copy()\n",
    "    d[\"hv\"] = haversine(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)\n",
    "    d[\"dist_fixed\"] = d.distance\n",
    "    d[\"ldist\"] = np.log(d.dist_fixed)\n",
    "    w = d.weight.abs().clip(upper=47500)\n",
    "    d[\"w_na\"] = w.isna().astype(int)\n",
    "    d[\"w\"] = w.fillna(weight_fill)\n",
    "    d[\"mi_na\"] = d.market_index.isna().astype(int)\n",
    "    d[\"mi\"] = d.market_index.fillna(d.date.map(mi_by_date)).fillna(mi_by_date.median())\n",
    "    d[\"eq\"] = d.equipment.map({e: i for i, e in enumerate(EQUIPMENT)})\n",
    "    d[\"dow\"] = d.date.dt.dayofweek\n",
    "    return d\n",
    "\n",
    "PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,\n",
    "              l2_regularization=1.0, random_state=0, categorical_features=CAT_IDX)\n",
    "HALF_LIFE_DAYS = 30  \n",
    "\n",
    "def fit_predict(train_df, test_df):\n",
    "    \"\"\"Train on train_df (with robust label cleaning + recency weights) and predict rates for test_df.\"\"\"\n",
    "    mi = pd.concat([train_df, test_df]).groupby(\"date\").market_index.median()\n",
    "    wfill = train_df.weight.abs().median()\n",
    "    a = make_features(train_df, mi, wfill)\n",
    "    b = make_features(test_df, mi, wfill)\n",
    "    y = np.log(a.posted_rate / a.dist_fixed)\n",
    "    weights = 0.5 ** ((a.date.max() - a.date).dt.days.values / HALF_LIFE_DAYS)\n",
    "    # pass 1: robust (absolute-error) model to spot corrupted targets\n",
    "    m0 = HistGradientBoostingRegressor(loss=\"absolute_error\", **PARAMS).fit(a[FEATURES], y, sample_weight=weights)\n",
    "    r = y - m0.predict(a[FEATURES])\n",
    "    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))\n",
    "    keep = (np.abs(r) < 4 * sigma).values\n",
    "    # pass 2: final model on cleaned labels\n",
    "    m = HistGradientBoostingRegressor(**PARAMS).fit(a.loc[keep, FEATURES], y[keep], sample_weight=weights[keep])\n",
    "    return np.exp(m.predict(b[FEATURES])) * b.dist_fixed.values, keep.mean()\n",
    "\n",
    "def metrics(y, p):\n",
    "    e = p - y\n",
    "    return {\"MAE\": np.abs(e).mean(), \"MAPE_%\": 100 * np.mean(np.abs(e) / y), \"medAPE_%\": 100 * np.median(np.abs(e) / y),\n",
    "            \"RMSE\": np.sqrt((e ** 2).mean()), \"R2\": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f7b8b65d",
   "metadata": {},
   "source": [
    "## 3. Validation strategy (time-based, no random split)\n",
    "Validation data is *future* data (Nov-Dec), so I mimic that with **expanding-window folds**: train on everything before the fold, test on the next two months\n",
    "(Jul-Aug and Sep-Oct). Metrics are reported on all rows (raw, includes corrupted labels which nobody can predict) and on \"inlier\" rows.\n",
    "A naive baseline (global median rate-per-mile × distance) shows the value of the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a82d3d06",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fold 2025-07: training rows kept after label cleaning = 95.9%\n",
      "fold 2025-09: training rows kept after label cleaning = 98.5%\n",
      "     test_window               setup     MAE  MAPE_%  medAPE_%    RMSE    R2\n",
      "2025-07..2025-09    model (all rows) 134.522   6.218     2.094 633.111 0.820\n",
      "2025-07..2025-09 model (inlier rows)  83.574   4.170     2.046 140.606 0.989\n",
      "2025-07..2025-09 baseline (all rows) 249.012  11.163     7.810 674.718 0.795\n",
      "2025-09..2025-11    model (all rows) 116.163   4.886     2.183 636.761 0.826\n",
      "2025-09..2025-11 model (inlier rows)  60.944   2.543     2.146  90.547 0.996\n",
      "2025-09..2025-11 baseline (all rows) 256.953  11.631     8.032 684.248 0.799\n"
     ]
    }
   ],
   "source": [
    "train = train.sort_values(\"date\").reset_index(drop=True)\n",
    "FOLDS = [(\"2025-07-01\", \"2025-09-01\"), (\"2025-09-01\", \"2025-11-01\")]\n",
    "rows = []\n",
    "for start, end in FOLDS:\n",
    "    A = train[train.date < start]\n",
    "    B = train[(train.date >= start) & (train.date < end)]\n",
    "    y = B.posted_rate.values\n",
    "    pred, kept = fit_predict(A, B)\n",
    "    base = B.distance.values * (A.posted_rate / A.distance).median()\n",
    "    rpm_b = (B.posted_rate / B.distance).values\n",
    "    inl = (rpm_b < 4) & (rpm_b > 0.9)\n",
    "    window = f\"{start[:7]}..{end[:7]}\"\n",
    "    rows.append({\"test_window\": window, \"setup\": \"model (all rows)\", **metrics(y, pred)})\n",
    "    rows.append({\"test_window\": window, \"setup\": \"model (inlier rows)\", **metrics(y[inl], pred[inl])})\n",
    "    rows.append({\"test_window\": window, \"setup\": \"baseline (all rows)\", **metrics(y, base)})\n",
    "    print(f\"fold {start[:7]}: training rows kept after label cleaning = {kept:.1%}\")\n",
    "print(pd.DataFrame(rows).round(3).to_string(index=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72f36269",
   "metadata": {},
   "source": [
    "## 4. Final model and validation-set predictions\n",
    "Train on **all** of `train_test.csv` with exactly the configuration validated above, then predict `validation.csv` in the template's order."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "47577947",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "final model trained; label rows kept after cleaning: 96.4%\n",
      "{'count': 12000.0, 'mean': 2356.8, 'std': 1367.6, 'min': 180.2, '25%': 1272.8, '50%': 2027.3, '75%': 3357.4, 'max': 6773.4}\n"
     ]
    }
   ],
   "source": [
    "val_pred, kept = fit_predict(train, valid)\n",
    "print(f\"final model trained; label rows kept after cleaning: {kept:.1%}\")\n",
    "\n",
    "out = pd.DataFrame({\"load_id\": valid.load_id, \"predicted_rate\": val_pred})\n",
    "out = template[[\"load_id\"]].merge(out, on=\"load_id\", how=\"left\")     \n",
    "assert len(out) == 12000 and out.load_id.is_unique\n",
    "assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()\n",
    "assert list(out.columns) == [\"load_id\", \"predicted_rate\"]\n",
    "out.to_csv(\"validation_predictions.csv\", index=False)\n",
    "print(out.predicted_rate.describe().round(1).to_dict())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c809b9c",
   "metadata": {},
   "source": [
    "## 5. December chart predictions\n",
    "The December file only gives route, distance, equipment, weight and date. The model also needs `market_index` and `quote_signal`, so they are taken from the\n",
    "validation loads on the *same calendar date* (daily median) rather than from a global training mean – the earlier version overwrote them with a constant.\n",
    "Coordinates come from the city lookup built from the data. The provided input file is **not** modified; results go to `december_predictions.csv`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b1373e61",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'count': 31.0, 'mean': 812.6, 'std': 5.4, 'min': 802.6, '25%': 807.8, '50%': 811.5, '75%': 817.7, 'max': 821.1}\n"
     ]
    }
   ],
   "source": [
    "cities = pd.concat([\n",
    "    train[[\"pickup\", \"pickup_lat\", \"pickup_lon\"]].set_axis([\"city\", \"lat\", \"lon\"], axis=1),\n",
    "    train[[\"delivery\", \"delivery_lat\", \"delivery_lon\"]].set_axis([\"city\", \"lat\", \"lon\"], axis=1),\n",
    "]).drop_duplicates(\"city\").set_index(\"city\")\n",
    "\n",
    "dec = dec_in.copy()\n",
    "dec[\"date\"] = pd.to_datetime(dec[\"date\"])\n",
    "daily = valid.groupby(\"date\").agg(market_index=(\"market_index\", \"median\"), quote_signal=(\"quote_signal\", \"median\"))\n",
    "dec = dec.join(daily, on=\"date\")\n",
    "assert dec[[\"market_index\", \"quote_signal\"]].notna().all().all(), \"no market data for some December dates\"\n",
    "dec[\"pickup_lat\"] = dec.pickup.map(cities.lat); dec[\"pickup_lon\"] = dec.pickup.map(cities.lon)\n",
    "dec[\"delivery_lat\"] = dec.delivery.map(cities.lat); dec[\"delivery_lon\"] = dec.delivery.map(cities.lon)\n",
    "\n",
    "dec_pred, _ = fit_predict(train, dec.drop(columns=\"predicted_rate\"))\n",
    "dec_out = dec_in.copy()\n",
    "dec_out[\"predicted_rate\"] = dec_pred\n",
    "dec_out.to_csv(\"december_predictions.csv\", index=False)\n",
    "print(dec_out.predicted_rate.describe().round(1).to_dict())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85a3e49d",
   "metadata": {},
   "source": [
    "## 6. Run the provided scorer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ec77ee6b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validated 12,000 final predictions.\n",
      "Validated 31 fixed December predictions.\n",
      "Created chart: scorer_results/candidate_december.png\n",
      "Final validation metrics are calculated by Spotter after submission.\n",
      " \n"
     ]
    }
   ],
   "source": [
    "res = subprocess.run([sys.executable, str(SCORER), \"--predictions\", \"validation_predictions.csv\",\n",
    "                      \"--december-predictions\", \"december_predictions.csv\"], capture_output=True, text=True)\n",
    "print(res.stdout, res.stderr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9f7c881e-997c-43bf-b9e5-0f8169e39d97",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (venv)",
   "language": "python",
   "name": "venv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
