# Build table takes the raw data and turns it into a workable formate for the project
# Author(s): Daniel
# Last Revised: 2026-11-13


from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw/SDC Platinum")

# raw data from sdc platinum split by year -> find all files
# column names are on excel row 3 (header = 2, index start at 0)
# drop blank rows from the excel export
files = sorted(RAW.glob("*.xlsx"))
df = pd.concat([pd.read_excel(f, header=2) for f in files], ignore_index=True)
df = df.dropna(subset=["SDC Deal No"])
assert df["SDC Deal No"].duplicated().sum() == 0, "Duplicate deal IDs"

# take columns that should be numeric
NUM = [
    "Percentage of Cash",
    "Initial Offer Price (USD)",
    "Target Share Price 4 Weeks Prior to Announcement (USD)",
    "Target Stock Price 1 Day After Announcement (USD)",
    "Deal Value (USD Millions)",
]
# text that can't be read as a number becomes NaN instead of crashing the script
df[NUM] = df[NUM].apply(pd.to_numeric, errors="coerce")

# run the check to make sure the data is clean
# run: .venv\Scripts\python.exe build_table.py
# expected output: 10 files, 2567 rows, 0 duplicate deal numbers
print(len(files), "files,", len(df), "rows,", df["SDC Deal No"].duplicated().sum(), "duplicate deal numbers")


# clean raw data

# isolate on 100% cash deals
# expected output: 10 files, 1153 rows, 0 duplicate deal numbers
df = df[df["Percentage of Cash"] == 100]
print(len(files), "files,", len(df), "rows,", df["SDC Deal No"].duplicated().sum(), "duplicate deal numbers")

# keep only deals where outcome is known
# expected output: 10 files, 1142 rows, 0 duplicate deal numbers
df = df[df["Deal Status"].isin(["Completed", "Withdrawn"])]
print(len(files), "files,", len(df), "rows,", df["SDC Deal No"].duplicated().sum(), "duplicate deal numbers")


# drop penny stocks
# expected output: 10 files, 1100 rows, 0 duplicate deal numbers
df = df[~(df["Target Share Price 4 Weeks Prior to Announcement (USD)"] < 1)]
print(len(files), "files,", len(df), "rows,", df["SDC Deal No"].duplicated().sum(), "duplicate deal numbers")

# label column where completed = 1 and withdrawn = 0
df["completion"] = (df["Deal Status"] == "Completed").astype(int)


# split column from announcement year

year = df["Date Announced"].dt.year
assert df.groupby(year)["completion"].min().eq(0).all(), "Every year must have a failure"
df["split"] = "train"
df.loc[year.between(2020, 2021), "split"] = "val"      # 2020-2021 -> validation
df.loc[year.between(2022, 2024), "split"] = "test"     # 2022-2024 -> test

# print split distribution
print(df["split"].value_counts())
# print split distribution for withdrawn deals
print(df[df["completion"] == 0]["split"].value_counts())


# Market implied probability (MIP)

offer = df["Initial Offer Price (USD)"]
prior_4_weeks = df["Target Share Price 4 Weeks Prior to Announcement (USD)"]
after_1_day = df["Target Stock Price 1 Day After Announcement (USD)"]

# denominator gets extremely small when offer is just a few cents above 4 week prior price
df["mip"] = (after_1_day - prior_4_weeks) / (offer - prior_4_weeks)

# Gap between offer and 4-week prior price must exceed 2% of the offer.
# Mark rows with missing day-1 prices as invalid.
df["mip_valid"] = (offer - prior_4_weeks > 0.02 * offer) & (df["mip"].notna())
# remove invalid MIP values
df.loc[~df["mip_valid"], "mip"] = float("nan")

# print number of valid MIP values
# Expected output: 978
print(df["mip_valid"].sum())

# Clip valid MIP to [0, 1] for Brier, retaining extreme deals and the original mip.
# Above-offer prices can reflect rival bids, so clipping is not probability calibration.
df["mip_prob"] = df["mip"].clip(0, 1)
for split in ["train", "val"]:
    split_mip = df.loc[df["split"].eq(split) & df["mip_valid"], "mip"]
    print(f"{split} valid MIP: above 1 = {(split_mip > 1).sum()}, below 0 = {(split_mip < 0).sum()}")

train_rate = df.loc[df["split"].eq("train"), "completion"].mean()
val = df.loc[df["split"].eq("val")]
print(f"Train completion rate: {train_rate:.4f}; base-rate Brier on all {len(val)} val deals: {((train_rate - val['completion']) ** 2).mean():.4f}")
# Compare both baselines only on valid-MIP val rows; exclude missing MIP from both.
val_mip = val.loc[val["mip_valid"]]
base_brier = ((train_rate - val_mip["completion"]) ** 2).mean()
mip_brier = ((val_mip["mip_prob"] - val_mip["completion"]) ** 2).mean()
assert 0 <= mip_brier <= 1
print(f"Validation Brier ({len(val_mip)} valid-MIP deals): base rate = {base_brier:.4f}, clipped MIP = {mip_brier:.4f}; lower is better")


# Feature flags

# label hostilities (and unsolicited)
# Expect 111 hostile/unsolicited deals
df["hostile"] = ((df["Deal Attitude"] == "Hostile") | (df["Deal Attitude"] == "Unsolicited")).astype(int)
print(df["hostile"].sum())

# turn tenders into a binary flag
# Expect 195 tender offers
df["tender"] = (df["Tender Offer Flag "] == True).astype(int)
print(df["tender"].sum())

# go-shop clause allows the target a time window to actively look for a better offer
# Expect 129 go-shop deals
df["go_shop"] = (df["Go Shop Flag"] == True).astype(int)
print(df["go_shop"].sum())

# label private equity acquisitions
# Expect 419 financial buyer deals
df["financial_buyer"] = ((df["Acquiror Firm Type"] == "FINANCIAL BUYER")).astype(int)
print(df["financial_buyer"].sum())

# label foreign acquirer acquisitions
# Expect 211 foreign acquirer deals
df["foreign_acquirer"] = (~(df["Acquiror Nation"] == "United States")).astype(int)
print(df["foreign_acquirer"].sum())

# label toehold: toehold is the share of the target the acquirer already owns before the deal
df["toehold"] = df["Percentage of Shares Held at Announcement"].fillna(0) / 100
print(df["toehold"].sum())

# label missing target termination fees
# Expect 402 missing target termination fees
df["target_fee_missing"] = df["Target Termination Fee (USD Millions)"].isna().astype(int)
print(df["target_fee_missing"].sum())

# label target termination fee as a percentage of the deal value
# Expect median termination fee percentage 0.029583589227764093
df["target_fee_pct"] = df["Target Termination Fee (USD Millions)"].fillna(0) / df["Deal Value (USD Millions)"]
print(df["target_fee_pct"].median())

# label missing acquiror termination fees
# Expect 722 missing acquiror termination fees
df["acquiror_fee_missing"] = df["Acquiror Termination Fee (USD Millions)"].isna().astype(int)
print(df["acquiror_fee_missing"].sum())

# label acquiror termination fee as a percentage of the deal value
# Expect median termination fee percentage 0.0
df["acquiror_fee_pct"] = df["Acquiror Termination Fee (USD Millions)"].fillna(0) / df["Deal Value (USD Millions)"]
print(df["acquiror_fee_pct"].median())

# premium = (offer / price 4 weeks before) - 1, too low or two high premiums - special circumstances
df["premium"] = df["Initial Offer Price (USD)"] / df["Target Share Price 4 Weeks Prior to Announcement (USD)"] - 1
train_premium = df.loc[df["split"] == "train", "premium"]
low, high = train_premium.quantile(0.01), train_premium.quantile(0.99)
df["premium"] = df["premium"].clip(lower=low, upper=high)

# record missing premiums
# Expect 66 missing premiums
df["premium_missing"] = df["premium"].isna().astype(int)
print(df["premium_missing"].sum())
df["premium"] = df["premium"].fillna(train_premium.median())

# label log deal value to compress extremes
df["log_deal_value"] = np.log(df["Deal Value (USD Millions)"] + 1)

# label same industry takeovers
# Expect 266 same industry takeovers
df["same_industry"] = (df["Target Primary SIC"] == df["Acquiror Primary SIC"]).astype(int)
print(df["same_industry"].sum())


# SIC flags
sic = pd.read_csv("data/sic1987_all.csv")
sic = sic.loc[sic["Level"] == 4].copy()   # 4 digit industry codes
assert len(sic) == 1005

# formate columns code -> int and title -> lowercase and stripped
sic["code_int"] = sic["code"].astype(int)
sic["title_clean"] = sic["title"].str.lower().str.strip()
target_sic_clean = df["Target Primary SIC"].str.lower().str.strip()

# two duplicate titles have alternative codes, both outside the regulated ranges
sic_lookup = dict(zip(sic["title_clean"], sic["code_int"]))
target_sic = target_sic_clean.map(sic_lookup)
print(f"Exact SIC matches: {target_sic_clean[target_sic.notna()].nunique()} names, {target_sic.notna().sum()} deals")

# Hand-mapped regulated-looking names; some turned out to fall outside the ranges.
sic_overrides = {
    "Accidental and health insurance": 6321,
    "Communications services, nec": 4899,
    "Health and allied services, nec": 8099,
    "Specialty outpatient facilities, nec": 8093,
    "State banks, member fed reserve": 6022,
    "Search, detection, and navigation equipment": 3812,
    "Cogeneration, alternative energy sources": 4911,
    "Aircraft parts, equipment": 3728,
    "Functions related to depository banking, nec": 6099,
    "Radio & TV broadcasting & communications equipment": 3663,
    "Telephone & telegraph apparatus": 3661,
    "Medical, dental, and hospital equipment & supplies": 5047,
    "Orthopedic, prosthetic, and surgical supplies": 3842,
    "X-Ray apparatus & tubes & other irradiation equip.": 3844,
    "Water, sewer, pipeline & utility line construction": 1623,
}
assert set(sic_overrides.values()).issubset(set(sic["code_int"]))
target_sic = target_sic.fillna(df["Target Primary SIC"].map(sic_overrides))
assert target_sic[df["Target Primary SIC"].isin(sic_overrides)].notna().all()

# industry code ranges
df["banks"] = (target_sic.between(6020, 6036) | (target_sic == 6712)).astype(int)
df["insurance"] = target_sic.between(6310, 6411).astype(int)
df["utilities"] = target_sic.between(4900, 4991).astype(int)
df["telecom_media"] = target_sic.between(4800, 4899).astype(int)
df["pharma_health"] = (target_sic.between(2834, 2836) | target_sic.between(8000, 8099)).astype(int)
df["defense"] = (target_sic.between(3760, 3769) | (target_sic == 3812)).astype(int)
df["airlines"] = (target_sic == 4512).astype(int)
SECTOR_FLAGS = ["banks", "insurance", "utilities", "telecom_media", "pharma_health", "defense", "airlines"]
print("Regulated-sector totals:")
print(df[SECTOR_FLAGS].sum())






# the prediction happens on day + 1 after announcement
# remove columns that wouldnt be known

# print(list(df.columns))

KEEP = [
    'SDC Deal No',
    'Date Announced',
    'Target Full Name',
    'Target 6-digit CUSIP',
    'Target PermID',
    'Target Primary Ticker Symbol',
    'Acquiror Full Name',
    'Acquiror PermID',
    'Acquiror Nation',
    'Acquiror Public Status',
    'Form of the Deal',
    'Percentage of Shares Acquiror is Seeking to Own after Transaction',
    'Percentage of Shares Acquiror is Seeking to Purchase in Transaction',
    'Percentage of Shares Held at Announcement',
    'Consideration Structure',
    'Percentage of Cash',
    'Initial Offer Price (USD)',
    'Target Share Price 4 Weeks Prior to Announcement (USD)',
    'Target Stock Price 1 Day After Announcement (USD)',
    'Deal Value (USD Millions)',
    'Target Primary SIC',
    'Acquiror Primary SIC',
    'Deal Attitude',
    'Sources of Funds',
    'Tender Offer Flag ',
    'Target Termination Fee (USD Millions)',
    'Acquiror Termination Fee (USD Millions)',
    'Acquiror Firm Type',
    'Go Shop Flag',
    'Rumored Deal Flag',
    'completion',
    'split',
    'mip',
    'mip_valid',
    'mip_prob',
    'hostile',
    'tender',
    'go_shop',
    'financial_buyer',
    'foreign_acquirer',
    'toehold',
    'target_fee_missing',
    'target_fee_pct',
    'acquiror_fee_missing',
    'acquiror_fee_pct',
    'premium',
    'premium_missing',
    'log_deal_value',
    'same_industry',

    ] + SECTOR_FLAGS

BANNED = [
    'Deal Status',
    'Date Effective',
    'Date Withdrawn',
    'Competing Offer Flag',
    'Competing Offer Acquiror Name',
    'Competing Offer SDC Deal Number',
    'Competing Offer Target Name',
    'Date Deal is Expected to be Effective',
    'Percentage Change Final Offer Price to Initial Offer Price',
    'Deal Synopsis'
]

leaked = set(BANNED) & set(KEEP)
assert not leaked, f"Leakage columns in output: {leaked}"


# save to CSV
Path("data").mkdir(exist_ok=True)
df = df[KEEP].copy()
FEATURES = KEEP[KEEP.index("hostile"):]
assert df[FEATURES].notna().all().all(), "Missing feature values"
assert np.isfinite(df[FEATURES].to_numpy(dtype=float)).all(), "Nonfinite feature values"
df.to_csv("data/deals.csv", index=False)
