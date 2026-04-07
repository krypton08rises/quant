import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from quant.data._common import SILVER_DIR, TripleBarrierSpec
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller

# Suppress warnings for cleaner output during research sprints
warnings.filterwarnings("ignore")


class SilverAuditor:
    def __init__(self, df: pd.DataFrame, target_col: str = "target_next_return"):
        """
        :param df: The Silver DataFrame (OHLCV + Indicators)
        :param target_col: The name of the column representing future returns (for IC analysis)
        """
        self.df = df.copy()
        self.target_col = target_col
        self.report = {}

        # Pre-processing: Ensure we have a proxy target for analysis if not present
        if self.target_col not in self.df.columns:
            # creating a simple 1-period forward log return for validation
            self.df[self.target_col] = np.log(self.df["close"].shift(-1) / self.df["close"])

        # Drop NaNs created by indicators/shifting for the sake of statistical tests
        self.df.dropna(inplace=True)

    def check_stationarity(self):
        """
        Runs Augmented Dickey-Fuller test on all numeric columns.
        Goal: p-value < 0.05
        """
        print("\n--- 1. STATIONARITY TEST (ADF) ---")
        results = []

        # specific focus on indicator columns, skipping metadata like 'symbol' or 'timestamp'
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col == self.target_col:
                continue

            try:
                # ADF Test
                adf_result = adfuller(self.df[col].values, autolag="AIC")
                p_value = adf_result[1]
                is_stationary = p_value < 0.05

                results.append(
                    {"Feature": col, "ADF P-Value": round(p_value, 4), "Stationary": is_stationary}
                )
            except Exception as e:
                print(f"Could not test {col}: {e}")

        self.stationarity_df = pd.DataFrame(results).sort_values(by="ADF P-Value", ascending=False)

        # Display the worst offenders (Highest P-Values)
        print(">> Top 5 Non-Stationary Features (Risk of drifting mean):")
        print(self.stationarity_df[~self.stationarity_df["Stationary"]].head(5))
        return self.stationarity_df

    def check_collinearity(self):
        """
        Checks correlation between features.
        Goal: Identify pairs with corr > 0.95
        """
        print("\n--- 2. COLLINEARITY CHECK ---")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Exclude target
        feature_cols = [c for c in numeric_cols if c != self.target_col]

        corr_matrix = self.df[feature_cols].corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        print(f">> Found {len(to_drop)} features with Correlation > 0.95.")
        if len(to_drop) > 0:
            print(f"   Examples: {to_drop[:5]} ...")

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()  # In a notebook, this displays inline. In script, saves or pops up.

    def check_signal_strength(self):
        """
        Calculates Information Coefficient (Spearman Corr) between feature and Target.
        Goal: Absolute IC > 0.02 is usually a starting point for daily data.
        """
        print("\n--- 3. PREDICTIVE POWER (Signal-to-Noise) ---")
        results = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col == self.target_col:
                continue

            # Spearman is better for non-linear relationships than Pearson
            corr, pval = spearmanr(self.df[col], self.df[self.target_col])

            results.append(
                {"Feature": col, "IC (Spearman)": round(corr, 4), "P-Value": round(pval, 4)}
            )

        self.ic_df = pd.DataFrame(results).sort_values(by="IC (Spearman)", key=abs, ascending=False)
        print(">> Top 5 Strongest Features (by raw correlation to next return):")
        print(self.ic_df.head(5))
        return self.ic_df


# --- USAGE MOCKUP ---
if __name__ == "__main__":
    # 1. Load your silver data (Mocking the load for this snippet)
    # df_silver = pd.read_parquet('data/silver/nifty50_silver.parquet')

    # FOR DEMONSTRATION: Creating dummy data to show functionality
    # np.random.seed(42)
    # dates = pd.date_range(start="2023-01-01", periods=500)
    # data = {
    #     'close': np.cumsum(np.random.randn(500)) + 100,
    #     'rsi': np.random.uniform(20, 80, 500), # Likely stationary
    #     'macd': np.cumsum(np.random.randn(500)), # Likely non-stationary (unbound)
    #     'ema_50': np.cumsum(np.random.randn(500)) + 100, # Highly correlated with close
    # }
    # df_silver = pd.DataFrame(data, index=dates)

    df_silver = pd.read_parquet(f"{SILVER_DIR}/day_{TripleBarrierSpec().config_str()}.parquet")

    # 2. Initialize Auditor
    auditor = SilverAuditor(df_silver)

    # 3. Run Checks
    auditor.check_stationarity()
    auditor.check_collinearity()
    auditor.check_signal_strength()
