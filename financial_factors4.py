import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

global target_vars
target_vars = [
    "ACTLCT",
    "APSALE",
    "CASHTA",
    "CHAT",
    "CHLCT",
    "EBITAT",
    "EBITSALE",
    "FAT",
    "FFOLT",
    "INVTSALES",
    "LCTLT",
    "LOGAT",
    "LOGSALE",
    "NIAT",
    "NIMTA",
    "NISALE",
    "TDEBT",
    "MKVAL",
    "LBTAT",
    "DBTAT",
    "DBTMKTEQ",
    "LBTMKTEQ",
]


def get_base_dataset():
    if os.path.exists("base_dataset.pkl"):
        print("Loading data from base_dataset.pkl...")
        with open("base_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    else:
        print("base_dataset.pkl not found. Running base_dataset3.py...")
        os.system("python base_dataset4.py")
        if os.path.exists("base_dataset.pkl"):
            with open("base_dataset.pkl", "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError("base_dataset.pkl not found after running base_dataset4.py")


def clean_dataset(df):
    df = df.loc[df["sector"] != "Financials"]
    df = df.loc[df["days2dflt"] >= 90]
    return df


def impute_data(df_initial2):
    df_initial2["act_est"] = df_initial2["che"] + df_initial2["rect"] + df_initial2["invt"]
    df_initial2["lct_est"] = df_initial2["ap"] + df_initial2["dlc"]
    df_initial2["act"] = np.where(
        df_initial2["act"].isna(), df_initial2["act_est"], df_initial2["act"]
    )
    df_initial2["lct"] = np.where(
        df_initial2["lct"].isna(), df_initial2["lct_est"], df_initial2["lct"]
    )

    df_initial2.drop(columns=["act_est", "lct_est"], inplace=True)

    df_initial2["xrd"] = df_initial2["xrd"].fillna(0)
    df_initial2["invt"] = df_initial2["invt"].fillna(0)

    return df_initial2


def build_features(df_initial3):
    # ACTLCT: Current Assets / Current Liabilities
    df_initial3["ACTLCT"] = df_initial3["act"] / df_initial3["lct"]

    # APSALE: Accounts Payable / Sales
    df_initial3["APSALE"] = df_initial3["ap"] / df_initial3["sale"]

    # CASHTA: Cash / Total Assets
    df_initial3["CASHTA"] = df_initial3["che"] / df_initial3["at"]

    # CHAT: Cash / Total Assets (duplicate for compatibility)
    df_initial3["CHAT"] = df_initial3["che"] / df_initial3["at"]

    # CHLCT: Cash / Current Liabilities
    df_initial3["CHLCT"] = df_initial3["che"] / df_initial3["lct"]

    # EBITAT: EBIT / Total Assets
    df_initial3["EBITAT"] = df_initial3["ebit"] / df_initial3["at"]

    # EBITSALE: EBIT / Sales
    df_initial3["EBITSALE"] = df_initial3["ebit"] / df_initial3["sale"]

    # FAT: (Short-term Debt + 0.5 × Long-term Debt) / Total Assets
    df_initial3["FAT"] = (df_initial3["dlc"] + 0.5 * df_initial3["dltt"]) / df_initial3["at"]

    # FFOLT: Operating Cash Flow / Total Liabilities
    df_initial3["FFOLT"] = df_initial3["oancf"] / df_initial3["lt"]

    # INVTSALES: Inventory / Sales
    df_initial3["INVTSALES"] = df_initial3["invt"] / df_initial3["sale"]

    # LCTLT: Current Liabilities / Total Liabilities
    df_initial3["LCTLT"] = df_initial3["lct"] / df_initial3["lt"]

    # LOGAT: log(Total Assets)
    df_initial3["LOGAT"] = df_initial3["at"].apply(lambda x: None if x <= 0 else np.log(x))

    # LOGSALE: log(Sales)
    df_initial3["LOGSALE"] = df_initial3["sale"].apply(lambda x: None if x <= 0 else np.log(x))

    # NIAT: Net Income / Total Assets
    df_initial3["NIAT"] = df_initial3["ni"] / df_initial3["at"]

    # NIMTA: Net Income / (Market Cap + Total Liabilities)
    df_initial3["NIMTA"] = df_initial3["ni"] / (
        df_initial3["prcc_f"] * df_initial3["csho"] + df_initial3["lt"]
    )

    # NISALE: Net Income / Sales
    df_initial3["NISALE"] = df_initial3["ni"] / df_initial3["sale"]

    # TDEBT: Total Debt = Short-term + Long-term Debt
    df_initial3["TDEBT"] = df_initial3["dlc"] + df_initial3["dltt"]

    # MKVAL: Market Capitalization = Price × Shares Outstanding
    df_initial3["MKVAL"] = df_initial3["prcc_f"] * df_initial3["csho"]

    # LBTAT: Total Liabilities / Total Assets
    df_initial3["LBTAT"] = df_initial3["lt"] / df_initial3["at"]

    # DBTAT: Total Debt / Total Assets
    df_initial3["DBTAT"] = df_initial3["TDEBT"] / df_initial3["at"]

    # DBTMKTEQ: Total Debt / (Total Debt + Market Cap or Equity)
    df_initial3["DBTMKTEQ"] = np.where(
        df_initial3["MKVAL"].notna(),
        df_initial3["TDEBT"] / (df_initial3["TDEBT"] + df_initial3["MKVAL"]),
        df_initial3["TDEBT"] / (df_initial3["TDEBT"] + df_initial3["ceq"]),
    )

    # LBTMKTEQ: Total Liabilities / (Total Liabilities + Market Cap or Equity)
    df_initial3["LBTMKTEQ"] = np.where(
        df_initial3["MKVAL"].notna(),
        df_initial3["lt"] / (df_initial3["lt"] + df_initial3["MKVAL"]),
        df_initial3["TDEBT"] / (df_initial3["TDEBT"] + df_initial3["ceq"]),
    )

    return df_initial3


def tobins_q_n_Altman_Z(df_base):
    df_base["datadate"] = pd.to_datetime(df_base["datadate"])

    df_base["ME"] = df_base["prcc_f"] * df_base["csho"]  # Market Value of Equity (ME)

    df_base["PREF"] = df_base[["ceq"]].fillna(0)  # Using Common Equity (ceq) as a proxy
    df_base["BE"] = df_base["ceq"] - df_base["PREF"]  # Book Value of Equity (BE)

    # Compute Tobin's Q
    df_base["Tobin_Q"] = (df_base["at"] + df_base["ME"] - df_base["BE"]) / df_base["at"]
    df_base.loc[df_base["at"] <= 0, "Tobin_Q"] = None  # Avoid invalid division

    # Compute Altman Z-Score
    df_base["Altman_Z"] = (
        3.3 * (df_base["ebit"] / df_base["at"])
        + 0.99 * (df_base["sale"] / df_base["at"])
        + 0.6 * (df_base["ME"] / df_base["lt"])
        + 1.2 * (df_base["act"] / df_base["at"])
        + 1.4 * (df_base["ni"] / df_base["at"])
    )

    # Fill NaNs only in numeric columns
    num_cols = df_base.select_dtypes(include=[np.number]).columns
    df_base[num_cols] = df_base[num_cols].fillna(0)


def calculate_auc(df_initial4):
    ac_scores = {}
    target_vars_2 = target_vars + ["Tobin_Q", "Altman_Z"]

    for var in target_vars_2:
        # Drop NaN rows for this variable and dflt_flag
        valid_rows = df_initial4[["dflt_flag", var]].dropna()
        if valid_rows["dflt_flag"].nunique() < 2:
            print(f"⚠️ Skipping {var}: only one class in dflt_flag after dropna.")
            continue

        auc = roc_auc_score(valid_rows["dflt_flag"], valid_rows[var])
        ac = abs(auc - 0.5) * 200  # 0–100 scale
        ac_scores[var] = ac

    # Sort and display
    for var, ac in sorted(ac_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{ac:6.2f}  \t {var}")


def get_final_dataframe():
    df = get_base_dataset()
    df = clean_dataset(df)
    df = impute_data(df)
    df = build_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=target_vars)
    tobins_q_n_Altman_Z(df)
    return df


def main():
    df = get_base_dataset()
    df = clean_dataset(df)
    df_imputed = impute_data(df)
    df_features = build_features(df_imputed)
    df_features_clean = df_features.replace([np.inf, -np.inf], np.nan).dropna(subset=target_vars)
    tobins_q_n_Altman_Z(df_features_clean)
    calculate_auc(df_features_clean)


if __name__ == "__main__":
    main()
