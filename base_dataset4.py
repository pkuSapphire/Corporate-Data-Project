# Description: This script downloads data from WRDS and processes it to create a base dataset for the credit risk model.
#
# Instructions:
# you may change conn = wrds.Connection() to your own WRDS credentials or use mine: wrds_username="bosen", wrds_password="Y@QC8f4qEjCz!eY"
# you don't need to rely on any external files, as the script will download the necessary data from GitHub and WRDS.

import subprocess
import sys

try:
    import wrds
except ImportError:
    print("Installing wrds package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wrds", "--quiet"])
    import wrds

import os
import io
import pandas as pd
import pickle
import requests
import datetime as dt

# Global WRDS connection
# WRDS_CONN = None
WRDS_CONN = wrds.Connection(wrds_username="bosen", wrds_password="Y@QC8f4qEjCz!eY")

# Global URLs for CSV downloads
major_groups_url = "https://raw.githubusercontent.com/saintsjd/sic4-list/master/major-groups.csv"
divisions_url = "https://raw.githubusercontent.com/saintsjd/sic4-list/master/divisions.csv"


def get_wrds_conn():
    global WRDS_CONN
    if WRDS_CONN is None:
        # Input credentials once; they will be reused in subsequent calls
        username = input("Enter WRDS username: ")  # Input WRDS username
        password = input("Enter WRDS password: ")  # Input WRDS password
        WRDS_CONN = wrds.Connection(wrds_username=username, wrds_password=password)
    return WRDS_CONN


def close_wrds_conn():
    global WRDS_CONN
    if WRDS_CONN is not None:
        WRDS_CONN.close()
        WRDS_CONN = None


def get_gvkey(filename="gvkey_data.pkl"):
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            gvkey = pickle.load(f)
    else:
        conn = get_wrds_conn()
        query = "SELECT * FROM ciq.wrds_gvkey"
        gvkey = conn.raw_sql(query)
        with open(filename, "wb") as f:
            pickle.dump(gvkey, f)
        print(f"Data saved to {filename}")
    return gvkey


def get_ratings(filename="ratings_data.pkl"):
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            ratings = pickle.load(f)
    else:
        conn = get_wrds_conn()
        query = """
        SELECT company_id as companyid, entity_pname, ratingdate, ratingsymbol, ratingactionword, unsol
        FROM ciq_ratings.wrds_erating
        WHERE longtermflag = 1 AND ratingtypename = 'Local Currency LT' AND ratingdate >= '1990-01-01'
        """
        ratings = conn.raw_sql(query)
        symbols = [
            "AAA",
            "AA+",
            "AA",
            "AA-",
            "A+",
            "A",
            "A-",
            "BBB+",
            "BBB",
            "BBB-",
            "BB+",
            "BB",
            "BB-",
            "B+",
            "B",
            "B-",
            "CCC+",
            "CCC",
            "CCC-",
            "CC",
            "C",
            "D",
            "SD",
            "NR",
            "R",
        ]
        ratings = ratings[ratings.ratingsymbol.isin(symbols)]
        with open(filename, "wb") as f:
            pickle.dump(ratings, f)
        print(f"Data saved to {filename}")
    return ratings


def merge_ratings_with_gvkey(gvkey, ratings):
    ratings2 = pd.merge(
        gvkey[["gvkey", "companyid", "startdate", "enddate"]], ratings, on="companyid"
    )
    ratings3 = ratings2.drop_duplicates(subset=["gvkey", "ratingdate"])
    ratings4 = ratings3.sort_values(
        ["gvkey", "companyid", "ratingdate"], ascending=[True, True, False]
    )
    ratings4["ratingenddate"] = ratings4.ratingdate.shift()
    ratings4["gvkey_"] = ratings4.gvkey.shift()
    ratings4.loc[ratings4.gvkey != ratings4.gvkey_, "ratingenddate"] = str(dt.date(2100, 12, 31))
    return ratings4


def get_sector(ratings4, filename="sector_data.pkl"):
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            info = pickle.load(f)
    else:
        conn = get_wrds_conn()
        sql_info = """
        SELECT
            gvkey,
            conm,
            fic,
            gsector,
            ggroup,
            gind,
            idbflag,
            incorp,
            loc,
            naics,
            sic,
            state
        FROM comp.company
        """
        info = conn.raw_sql(sql_info)
        info = info[info.gvkey.isin(ratings4.gvkey)]
        info = info.drop_duplicates(subset=["gvkey"])
        with open(filename, "wb") as f:
            pickle.dump(info, f)
        print(f"Data saved to {filename}")
    return info


def get_or_download_csv(pkl_filename, csv_url):
    if os.path.exists(pkl_filename):
        print(f"Loading data from {pkl_filename}...")
        df = pd.read_pickle(pkl_filename)
    else:
        print(f"{pkl_filename} not found, downloading from GitHub...")
        response = requests.get(csv_url)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df.to_pickle(pkl_filename)
            print(f"Data saved to {pkl_filename}")
        else:
            raise Exception(f"Download failed with status code: {response.status_code}")
    return df


def get_sector_info(ratings4):
    info = get_sector(ratings4)
    major_groups = get_or_download_csv("major_groups.pkl", major_groups_url)
    divisions = get_or_download_csv("divisions.pkl", divisions_url)
    info["Major Group"] = info["sic"].astype(str).str[:2].str.zfill(2)
    major_groups["Major Group"] = major_groups["Major Group"].astype(str).str.zfill(2)
    info_with_division = info.merge(
        major_groups[["Major Group", "Division"]], on="Major Group", how="left"
    )
    info_with_division = info_with_division.merge(
        divisions[["Division", "Description"]], on="Division", how="left"
    )
    info_with_division.rename(columns={"Description": "SIC Division Name"}, inplace=True)
    info_with_division = info_with_division.drop(columns=["Major Group", "Division"])

    gics_data = {
        "gsector": ["10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"],
        "GIC Sector Name": [
            "Energy",
            "Materials",
            "Industrials",
            "Consumer Discretionary",
            "Consumer Staples",
            "Health Care",
            "Financials",
            "Information Technology",
            "Communication Services",
            "Utilities",
            "Real Estate",
        ],
    }
    gics = pd.DataFrame(gics_data)
    info_with_gic = info_with_division.merge(gics, on="gsector", how="left")
    info_with_gic["sector"] = info_with_gic["GIC Sector Name"]
    info_with_gic.loc[info_with_gic["sector"].isnull(), "sector"] = info_with_gic[
        "SIC Division Name"
    ]
    info_1 = info_with_gic.drop(columns=["fic", "gind", "idbflag", "incorp", "state"])

    # Adjust sectors for specific companies
    info_1.loc[info_1["conm"] == "ARGO GROUP INTL 6.5 SR NT 42", "sector"] = "Insurance"
    info_1.loc[info_1["conm"] == "HILFIGER (TOMMY) U S A INC", "sector"] = "Manufacturing"
    info_1.loc[info_1["conm"] == "NOVA SCOTIA POWER INC", "sector"] = "Utilities"

    def replace_sector(row):
        if row["sector"] in ["Consumer Discretionary", "Consumer Staples"]:
            return row["SIC Division Name"]
        else:
            return row["sector"]

    info_2 = info_1.copy()
    info_2["sector"] = info_2.apply(replace_sector, axis=1)
    info_2["ggroup"] = info_2["ggroup"].fillna("")

    sector_mapping = {
        "Industrials": "Manufacturing",
        "Health Care": "Health",
        "Energy": "Utilities",
        "Information Technology": "Information Technology",
        "Wholesale Trade": "Wholesale",
        "Utilities": "Utilities",
        "Financials": "Financials",
        "Materials": "Manufacturing",
        "Transportation, Communications, Electric, Gas, And Sanitary Services": "Transportation, Communications, Electric, Gas, And Sanitary Services",
        "Communication Services": "Services",
        "Retail Trade": "Retail",
        "Manufacturing": "Manufacturing",
        "Construction": "Construction",
        "Finance, Insurance, And Real Estate": "Financials",
        "Services": "Services",
        "Agriculture, Forestry, And Fishing": "Agriculture",
        "Public Administration": "Utilities",
        "Real Estate": "Financials",
        "Mining": "Manufacturing",
        "Insurance": "Financials",
    }
    info_3 = info_2.copy()
    info_3["sector"] = info_3["sector"].map(sector_mapping)

    def map_sic_to_sector(row):
        if row["sector"] == "Transportation, Communications, Electric, Gas, And Sanitary Services":
            sic_major_group = int(str(row["sic"])[:2])
            if 40 <= sic_major_group < 48:
                return "Transportation"
            elif sic_major_group == 48:
                return "Services"
            else:
                return "Utilities"
        else:
            return row["sector"]

    def map_gic_transportation(row):
        if row["ggroup"] == "2030":
            return "Transportation"
        else:
            return row["sector"]

    info_3["sector"] = info_3.apply(map_sic_to_sector, axis=1)
    info_3["sector"] = info_3.apply(map_gic_transportation, axis=1)
    return info_3


def prepare_ratings(info_3, ratings4):
    ratings5 = pd.merge(ratings4, info_3[["gvkey", "sector"]], on="gvkey", how="left")
    ratings_all = ratings5.copy()
    defaults_all = ratings_all[ratings_all.ratingsymbol.isin(["D", "SD", "R"])].copy()
    defaults_all2 = defaults_all[["gvkey", "ratingdate"]].drop_duplicates("gvkey")
    defaults_all2["default_flag"] = 1
    ratings6 = pd.merge(ratings5, defaults_all2, on=["gvkey", "ratingdate"], how="left")
    ratings6.loc[pd.isnull(ratings6.default_flag), "default_flag"] = 0
    return ratings6


def get_financials(filename="financials_data.pkl"):
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            financials = pickle.load(f)
    else:
        conn = get_wrds_conn()
        sql_financials = """
        SELECT
            gvkey,
            datadate,
            fyear,
            fyr,
            at,
            lt,
            ceq,
            act,
            lct,
            invt,
            rect,
            ap,
            dlc,
            dltt,
            dltis,
            dvt,
            che,
            xint,
            xrd,
            xsga,
            oibdp,
            ebit,
            sale,
            cogs,
            ni,
            oancf,
            fincf,
            csho,
            prcc_f,
            'Annual' AS freq
        FROM comp.funda
        WHERE
            indfmt = 'INDL'
            AND datafmt = 'STD'
            AND popsrc = 'D'
            AND consol = 'C'
            AND fyear >= 1990
        """
        financials = conn.raw_sql(sql_financials)
        financials.to_pickle(filename)
        print(f"Data saved to {filename}")
    return financials


def prepare_financials():
    financials = get_financials()
    financials = financials.sort_values(by=["gvkey", "datadate"], ascending=[True, False])
    return financials


def merge_financials_ratings(financials, ratings6):
    common_gvkeys = set(financials["gvkey"]).intersection(set(ratings6["gvkey"]))
    financials2 = financials[financials["gvkey"].isin(common_gvkeys)].copy()
    ratings7 = ratings6[ratings6["gvkey"].isin(common_gvkeys)].copy()
    financials2["datadate"] = pd.to_datetime(financials2["datadate"], errors="coerce")
    ratings7["ratingdate"] = pd.to_datetime(ratings7["ratingdate"], errors="coerce")
    ratings7["ratingenddate"] = pd.to_datetime(ratings7["ratingenddate"], errors="coerce")
    ratings7.loc[ratings7["ratingenddate"] > "2100-12-31", "ratingenddate"] = pd.Timestamp(
        "2100-12-31"
    )
    merged_df = financials2.merge(ratings7, how="left", on="gvkey")
    merged_df["datadate"] = pd.to_datetime(merged_df["datadate"], errors="coerce")
    merged_df["ratingdate"] = pd.to_datetime(merged_df["ratingdate"], errors="coerce")
    merged_df["ratingenddate"] = pd.to_datetime(merged_df["ratingenddate"], errors="coerce")
    merged_df = merged_df[
        (merged_df["ratingdate"].notna())
        & (merged_df["ratingenddate"].notna())
        & (merged_df["ratingdate"] <= merged_df["datadate"])
        & (merged_df["datadate"] <= merged_df["ratingenddate"])
    ]
    merged_df = merged_df.sort_values(by=["gvkey", "datadate"], ascending=[True, True])
    merged_df.reset_index(drop=True, inplace=True)
    columns_to_keep = list(financials2.columns) + [
        "entity_pname",
        "ratingdate",
        "ratingsymbol",
        "ratingactionword",
        "unsol",
        "ratingenddate",
        "sector",
    ]
    mfinancials_df = merged_df[columns_to_keep]
    mfinancials_df = override_by_exact_fyear(mfinancials_df, ratings6)

    return mfinancials_df


def compute_default_dates(mfinancials_df):
    default_date_df = mfinancials_df.filter(["gvkey", "ratingsymbol", "ratingdate"]).copy()

    def find_default_date(row):
        if row["ratingsymbol"] in ["D", "SD", "R"]:
            return row["ratingdate"]
        else:
            return pd.NaT

    default_date_df["dflt_date"] = default_date_df.apply(find_default_date, axis=1)
    default_date_df["dflt_date"] = pd.to_datetime(default_date_df["dflt_date"])
    default_date_df = default_date_df.sort_values(by=["gvkey", "dflt_date"], ascending=[True, True])
    default_date_df = default_date_df.groupby("gvkey", as_index=False).first()
    default_date_df["dflt_date"] = default_date_df["dflt_date"].fillna(pd.Timestamp("2100-12-31"))
    default_date_df = default_date_df.drop(["ratingdate", "ratingsymbol"], axis=1)
    default_date_df["dflt_date"] = pd.to_datetime(default_date_df["dflt_date"])
    return default_date_df


def merge_default_dates(mfinancials_df, default_date_df):
    df1 = pd.merge(mfinancials_df, default_date_df, on=["gvkey"], how="left")
    df2 = df1.copy()
    df2["dflt_date"] = pd.to_datetime(df2["dflt_date"], errors="coerce")
    df2["datadate"] = pd.to_datetime(df2["datadate"], errors="coerce")
    df2["days2dflt"] = (df2["dflt_date"] - df2["datadate"]).dt.days
    df3 = df2.copy()
    df3["dflt_flag"] = 0
    df3.loc[(df3["days2dflt"] >= 90) & (df3["days2dflt"] <= 455), "dflt_flag"] = 1
    return df3


def clean_dataset(df):
    # Use sector to exclude financial institutions, insurance companies, and real estate firms
    df = df.loc[df["sector"] != "Financials"]
    # Use days2dflt to exclude future information
    df = df.loc[df["days2dflt"] >= 90]
    return df


def check_missing_financials_vs_ratings(ratings6, financials):
    ratings_gvkey = set(ratings6["gvkey"])
    financials_gvkey = set(financials["gvkey"])
    only_in_ratings = ratings_gvkey - financials_gvkey
    missing_info = ratings6[ratings6["gvkey"].isin(only_in_ratings)]
    missing_info[["gvkey", "entity_pname"]].drop_duplicates().to_csv(
        "missing_financials_gvkeys.csv", index=False
    )


def override_by_exact_fyear(mfinancials_df, ratings6):
    defaults = ratings6[ratings6["ratingsymbol"].isin(["D", "SD", "R"])][
        ["gvkey", "ratingdate", "ratingsymbol"]
    ].copy()
    defaults["ratingdate"] = pd.to_datetime(defaults["ratingdate"], errors="coerce")
    defaults["fyear"] = defaults["ratingdate"].dt.year

    overrides = (
        defaults.sort_values("ratingdate")
        .drop_duplicates(subset=["gvkey", "fyear"])
        .rename(
            columns={"ratingsymbol": "ratingsymbol_override", "ratingdate": "ratingdate_override"}
        )
    )

    df = pd.merge(
        mfinancials_df,
        overrides[["gvkey", "fyear", "ratingsymbol_override", "ratingdate_override"]],
        on=["gvkey", "fyear"],
        how="left",
    )

    df["ratingsymbol"] = df["ratingsymbol_override"].combine_first(df["ratingsymbol"])
    df["ratingdate"] = df["ratingdate_override"].combine_first(df["ratingdate"])

    df.drop(columns=["ratingsymbol_override", "ratingdate_override"], inplace=True)

    return df


def main():
    # Load gvkey and ratings data and merge them
    gvkey = get_gvkey()
    ratings = get_ratings()
    ratings4 = merge_ratings_with_gvkey(gvkey, ratings)

    # Process sector information and prepare ratings
    info_3 = get_sector_info(ratings4)
    ratings6 = prepare_ratings(info_3, ratings4)

    # Load financials and merge with ratings
    financials = prepare_financials()

    # Check missing gvkeys in financials vs. ratings
    # check_missing_financials_vs_ratings(ratings6, financials)

    mfinancials_df = merge_financials_ratings(financials, ratings6)
    close_wrds_conn()

    # Compute default dates and merge default flags
    default_date_df = compute_default_dates(mfinancials_df)
    all_df = merge_default_dates(mfinancials_df, default_date_df)

    # clean data
    # final_df = clean_dataset(all_df)
    final_df = all_df

    print(final_df.describe())
    final_df.to_pickle("base_dataset.pkl")
    print("Data saved to base_dataset.pkl")


if __name__ == "__main__":
    main()
