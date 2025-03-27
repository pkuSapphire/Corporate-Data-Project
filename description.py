import os
import pickle
import financial_factors4 as ff4


def get_base_dataset():
    # Check if base_dataset.pkl exists
    if os.path.exists("base_dataset.pkl"):
        print("Loading data from base_dataset.pkl...")
        with open("base_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    else:
        # Run base_dataset3.py to create base_dataset.pkl
        print("base_dataset.pkl not found. Running base_dataset3.py...")
        os.system("python base_dataset3.py")
        if os.path.exists("base_dataset.pkl"):
            with open("base_dataset.pkl", "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError("base_dataset.pkl not found after running base_dataset3.py")


def clean_dataset(df):
    # Use sector to exclude financial institutions, insurance companies, and real estate firms
    # print("Original dataset:", len(df))
    df = df.loc[df["sector"] != "Financials"]
    # print("After removing Financials:", len(df))
    # Use days2dflt to exclude future information
    df = df.loc[df["days2dflt"] >= 90]
    # print("After filtering days2dflt >= 90:", len(df))
    return df


# Description of Dataset


# Table 1: Sample Data Information
# A table shows PERIOD FIRMS DEFAULTS STATEMENTS
def sample_data_info():
    df = get_base_dataset()
    # print(df.head())
    # print(df.describe())


# Figure 1: Distribution of Statements and Defaults by Year


def statements_defaults_by_year():
    df = get_base_dataset()
    df = clean_dataset(df)
    df["year"] = df["fyear"].astype(int)

    # Group by year: count total firms and sum defaults
    summary = (
        df.groupby("year")
        .agg(total_firms=("dflt_flag", "count"), total_defaults=("dflt_flag", "sum"))
        .reset_index()
    )

    summary["default_rate"] = summary["total_defaults"] / summary["total_firms"]

    summary = (
        df.groupby("year")
        .agg(total_firms=("dflt_flag", "count"), total_defaults=("dflt_flag", "sum"))
        .reset_index()
    )

    # shift defaults and default_rate by one year
    summary["defaults_year"] = summary["year"] + 1
    summary["default_rate"] = summary["total_defaults"] / summary["total_firms"]

    # filter to only include data where defaults_year <= 2022
    summary = summary[summary["year"] <= 2022]

    return summary


import matplotlib.pyplot as plt


def plot_statements_and_defaults_dual_axis():
    summary = statements_defaults_by_year()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(
        summary["year"], summary["total_firms"], color="skyblue", label="Total Firms", alpha=0.7
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Firms")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(
        summary["defaults_year"], summary["total_defaults"], "ro-", label="Defaults", markersize=6
    )
    ax2.set_ylabel("Defaults")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Total Firms and Defaults by Year (Defaults Shifted +1 Year)")
    plt.tight_layout()
    plt.show()


def statements_defaults_by_industry():
    df = get_base_dataset()
    df = clean_dataset(df)

    summary = (
        df.groupby("sector")
        .agg(
            total_firms=("gvkey", lambda x: x.nunique()),
            total_defaults=("dflt_flag", "sum"),
        )
        .reset_index()
    )

    summary["default_rate"] = summary["total_defaults"] / summary["total_firms"]
    summary = summary.sort_values(by="default_rate", ascending=False)

    return summary


def plot_statements_defaults_by_industry():
    summary = statements_defaults_by_industry()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(summary["sector"], summary["total_firms"], color="lightblue", label="Total Firms")
    ax1.set_ylabel("Total Firms")
    ax1.set_xlabel("Industry Sector")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, axis="y")

    target_height = summary["total_firms"].max() * 0.6
    scaling_factor = target_height / summary["default_rate"].max()
    scaled_default_rate = summary["default_rate"] * scaling_factor
    ax1.plot(
        summary["sector"], scaled_default_rate, "k--", label="Default Rate (scaled)", linewidth=2
    )

    for x, y, r in zip(summary["sector"], scaled_default_rate, summary["default_rate"]):
        ax1.text(
            x,
            y + 0.01 * target_height,
            f"{r:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax2 = ax1.twinx()
    ax2.plot(summary["sector"], summary["total_defaults"], "ro-", label="Defaults", markersize=6)
    ax2.set_ylabel("Number of Defaults")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title(f"Industries by Cumulative Default Rate")
    plt.tight_layout()
    plt.show()


def main():
    # table 1
    sample_data_info()
    # figure 1
    # plot_statements_defaults()
    plot_statements_and_defaults_dual_axis()
    # figure 2
    plot_statements_defaults_by_industry()
    df = ff4.get_final_dataframe()
    ff4.calculate_auc(df)


if __name__ == "__main__":
    main()
