# %%
import financial_factors4 as ff4
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


def evaluate_single_var_model(df, var, target="dflt_flag"):
    df = df[[var, target]].dropna()
    X = df[[var]]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\nSingle-Variable Model: {var}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score, fpr, tpr


def evaluate_multivariate_model(df, feature_cols, target="dflt_flag", model_label="Multivariate"):
    df = df[feature_cols + [target]].dropna()
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\n{model_label}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score, fpr, tpr


def evaluate_l1_model(df, feature_cols, target="dflt_flag"):
    df = df[feature_cols + [target]].dropna()
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\nL1-Penalized Logistic Regression")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Coefficients:\n", clf.coef_)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    non_zero_coef = np.sum(clf.coef_ != 0)
    print(f"Number of non-zero features selected: {non_zero_coef} / {X.shape[1]}")

    return auc_score, fpr, tpr


def plot_roc_curves(roc_data):
    plt.figure(figsize=(6, 4))
    for label, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.2f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    df = ff4.get_final_dataframe()

    auc_tobin, fpr_tobin, tpr_tobin = evaluate_single_var_model(df, "Tobin_Q")
    auc_altman, fpr_altman, tpr_altman = evaluate_single_var_model(df, "Altman_Z")

    all_vars = ["Tobin_Q", "Altman_Z"]
    auc_combo, fpr_combo, tpr_combo = evaluate_multivariate_model(
        df, all_vars, model_label="Combined Model"
    )

    auc_ours, fpr_ours, tpr_ours = evaluate_multivariate_model(
        df, ff4.target_vars, model_label="Our Model"
    )

    auc_l1, fpr_l1, tpr_l1 = evaluate_l1_model(df, ff4.target_vars)

    print("\nSummary AUCs:")
    print(f"Model A - Tobin's Q         AUC: {auc_tobin:.4f}")
    print(f"Model B - Altman Z          AUC: {auc_altman:.4f}")
    print(f"Model C - Combined All      AUC: {auc_combo:.4f}")
    print(f"Model D - Our Factors       AUC: {auc_ours:.4f}")
    print(f"Model E - L1-Regularized    AUC: {auc_l1:.4f}")

    roc_data = {
        "Tobin_Q": (fpr_tobin, tpr_tobin, auc_tobin),
        "Altman_Z": (fpr_altman, tpr_altman, auc_altman),
        "Combined (All)": (fpr_combo, tpr_combo, auc_combo),
        "Our Model": (fpr_ours, tpr_ours, auc_ours),
        "L1-Penalized": (fpr_l1, tpr_l1, auc_l1),
    }

    plot_roc_curves(roc_data)


# %%
if __name__ == "__main__":
    main()

# %%
