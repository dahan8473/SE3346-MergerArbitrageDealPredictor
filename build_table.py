# Build table takes the raw data and turns it into a workable formate for the project
# Author(s): Daniel
# Last Revised: 2026-11-12


from pathlib import Path
import pandas as pd

RAW = Path("data/raw/SDC Platinum")

# raw data from sdc platinum split by year -> find all files
# column names are on excel row 3 (header = 2, index start at 0)
# drop blank rows from the excel export
files = sorted(RAW.glob("*.xlsx")) 
df = pd.concat([pd.read_excel(f, header=2) for f in files], ignore_index=True)
df = df.dropna(subset=["SDC Deal No"])

# take needed columns from the dataframe
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
df["label"] = (df["Deal Status"] == "Completed").astype(int)


# split column from announcement year

year = df["Date Announced"].dt.year
df["split"] = "train"
df.loc[year.between(2020, 2021), "split"] = "val"      # 2020-2021 -> validation
df.loc[year.between(2022, 2024), "split"] = "test"     # 2022-2024 -> test

# print split distribution
print(df["split"].value_counts())
# print split distribution for withdrawn deals
print(df[df["label"] == 0]["split"].value_counts())


# Market implied probability (MIP)

offer = df["Initial Offer Price (USD)"]
prior_4_weeks = df["Target Share Price 4 Weeks Prior to Announcement (USD)"]
after_1_day = df["Target Stock Price 1 Day After Announcement (USD)"]

# denominator gets extremely small when offer is just a few cents above 4 week prior price
df["mip"] = (after_1_day - prior_4_weeks) / (offer - prior_4_weeks)   

# bap b/w offer and 4 week prior price should be > 2% of 4 week prior price
# drop found lines where price 1 day after is missing
df["mip_valid"] = (offer - prior_4_weeks > 0.02 * offer) & (df["mip"].notna())
# remove invalid MIP values
df.loc[~df["mip_valid"], "mip"] = float("nan")  

# print number of valid MIP values
# Expected output: 978
print(df["mip_valid"].sum())


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
    'Collar Flag', 
    'Rumored Deal Flag', 
    'label', 
    'split', 
    'mip', 
    'mip_valid'
    ]

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
df[KEEP].to_csv("data/deals.csv", index=False)






