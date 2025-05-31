import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import johnsonsu


def safe_divide(x, y):
    if y == 0:
        return np.nan
    return x / y


def multiplier(
    a_crisis,
    a_normal,
    b_crisis,
    b_normal,
    loc_crisis,
    loc_normal,
    scale_crisis,
    scale_normal,
):
    a_multiplier = safe_divide(a_crisis, a_normal)
    b_multiplier = safe_divide(b_crisis, b_normal)
    loc_multiplier = safe_divide(loc_crisis, loc_normal)
    scale_multiplier = safe_divide(scale_crisis, scale_normal)

    return a_multiplier, b_multiplier, loc_multiplier, scale_multiplier


def plot(x, pdf_normal, pdf_crisis):
    plt.figure(figsize=(12, 6))
    plt.plot(x, pdf_normal, label="Normal period", color="blue")
    plt.plot(x, pdf_crisis, label="Crisis period", color="red")
    plt.title("Johnson SU :Norma vs Crisis period Daily Rate of Return")
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_johnson_su_distribution(file_path):
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    returns = df["Close"].pct_change().dropna()

    normal_period = returns["2012-09-15":"2015-01-13"]
    crisis_period = returns["2006-09-15":"2009-01-13"]

    params_normal = johnsonsu.fit(normal_period)
    params_crisis = johnsonsu.fit(crisis_period)

    print("Normal period Johnson SU ：", tuple(f"{p:.4f}" for p in params_normal))
    print("Crisis period Johnson SU ：", tuple(f"{p:.4f}" for p in params_crisis))

    a_normal, b_normal, loc_normal, scale_normal = params_normal
    a_crisis, b_crisis, loc_crisis, scale_crisis = params_crisis

    x = np.linspace(-0.1, 0.1, 1000)

    pdf_normal = johnsonsu.pdf(x, *params_normal)
    pdf_crisis = johnsonsu.pdf(x, *params_crisis)

    plot(x, pdf_normal, pdf_crisis)

    a_multiplier, b_multiplier, loc_multiplier, scale_multiplier = multiplier(
        a_crisis,
        a_normal,
        b_crisis,
        b_normal,
        loc_crisis,
        loc_normal,
        scale_crisis,
        scale_normal,
    )

    print("\nParameter change multiple：")
    print(f"a：{a_multiplier:.2f} times")
    print(f"b：{b_multiplier:.2f} times")
    print(f"loc：{loc_multiplier:.2f} times")
    print(f"scale：{scale_multiplier:.2f} times")


def main():
    file_path = "data/KLCI_20y.csv"
    analyze_johnson_su_distribution(file_path)


if __name__ == "__main__":
    txn = main()
