Project: Automated Stock Forecasting & Trading Pipeline
Overview

This project leverages quantitative stock data from yfinance and employs the NBEATSx model from nixtla for forecasting. The end goal is to develop a conservative automated trading system that identifies stocks with a high probability of significant price movement over a week. So far, the pipeline includes:

    Data fetching and feature engineering (e.g., calculating candlestick features, percent change, log-scaled volume, etc.).

    Rolling window backtesting using a sliding window approach (50 days of history predicting the next 5 days).

    Visualization of predictions versus actuals along with error distributions.

    Basic logging and experiment tracking through generated logs and plots.

What Has Been Done

    Data Pipeline:

        Fetching stock data from Yahoo Finance using yfinance.

        Preprocessing including normalization, feature creation (e.g., upper_shadow, lower_shadow, tick_body), and scaling.

    Modeling:

        Implementation of the NBEATSx model wrapped within a rolling forecast evaluation framework.

        Generation of backtest plots to visually compare predictions with actual values.

        Calculation of error metrics (MAE, RMSE) per stock and across rolling windows.

    Visualization & Logging:

        Plotting true vs. predicted values and error distributions.

        Logging of experiment progress and errors in designated log folders.

Roadmap & Next Steps
1. Automating Experimentation

Rather than running experiments manually, it’s best to automate as many steps as possible. Automation helps with reproducibility, efficient tracking, and rapid iteration. Here’s how to proceed:

    Build a Modular Pipeline:

        Data Ingestion Module: Automate data fetching, normalization, and feature engineering.

        Model Training Module: Set up scripts to run training and evaluation with varying parameters.

        Evaluation Module: Automate the generation of plots and error metrics calculation.

    Experiment Orchestration:

        Use a tool like MLflow, Sacred, or Weights & Biases for experiment tracking.

        Design your experiments so that you can vary hyperparameters, time horizons, and features through configuration files or command-line arguments.

    Scheduling & CI/CD:

        Use a task scheduler (like cron or Airflow) to run experiments on a set schedule, ensuring that new data or configurations are picked up automatically.

        Integrate version control (Git) to track changes in both code and experiment configurations.

2. Establishing a Baseline

    Initial Baseline Model:

        Run your current pipeline with a fixed set of hyperparameters (e.g., input_size=50, window_size=5, specific candlestick features) to establish baseline error metrics (MAE, RMSE) and directional accuracy.

        Document these baseline metrics clearly as a reference for future experiments.

    Reproducibility:

        Use random seeds (as already done) and document the environment details to ensure the baseline is reproducible.

3. Metrics Calculation & Analysis

    Error Metrics:

        Continue using MAE and RMSE. Additionally, consider directional accuracy, hit ratios (how often the predicted direction matches the actual), and profit-related metrics once you simulate trades.

    Visualization:

        Automate the generation of time-series plots, error distribution charts, and summary CSVs that aggregate metrics across rolling windows.

    Comparative Analysis:

        Build scripts to compare the performance of different experimental setups. For example, a summary dashboard or report that lists each experiment’s configuration and results side-by-side.

4. Hyperparameter Tuning

    Automated Hyperparameter Search:

        Integrate hyperparameter tuning libraries (e.g., Optuna, Hyperopt) to search over key parameters such as learning rate, window size, and number of training steps.

        Log each configuration and its performance automatically.

    Experiment Tracking:

        Save model configurations, performance metrics, and visualizations for each run.

        Use the experiment tracking tool to quickly compare the impact of different hyperparameter settings.

5. Experimenting with Time Horizons & Feature Engineering

    Vary Time Horizons:

        Design experiments to test different forecast windows (e.g., 3, 5, 7, or 10 days) to find the most reliable interval.

        Automate rolling window evaluations for each time horizon setting.

    Feature Engineering:

        Test the impact of different sets of features (technical indicators, candlestick features, volume transformations, etc.).

        Run controlled experiments where you add or remove features, and document how each affects forecast accuracy and trading signal reliability.

6. Documentation & Tracking

    Maintain a Central Document:

        Use this README (or a dedicated project notebook) as the central documentation hub.

        Create a section for Experiment Log where you record:

            Date and experiment ID.

            Configuration (hyperparameters, features, time horizon).

            Performance metrics and qualitative observations.

            Visual artifacts (plots, error distribution charts).

    Versioning Experiments:

        Use a consistent naming scheme and version control for experiment configurations.

        Consider using Jupyter notebooks or a lightweight wiki if you prefer a more interactive documentation style.

Proposed Folder Structure

/project-root
│
├── data/                # Data files and stock tickers
├── logs/                # Logs for data fetching, model training, and experiments
├── plots/               # Generated plots from evaluations
├── experiments/         # Config files and summaries for each experiment
├── src/                 # Source code for data ingestion, modeling, evaluation, etc.
│   ├── fetch_data.py
│   ├── nbeatsx.py
│   └── experiment_runner.py  # Script to automate experiments (with CLI options)
├── README.md            # This roadmap & documentation file
└── requirements.txt     # Python dependencies

Final Recommendations

    Automation is Key:
    Automate your experiments as much as possible. This not only reduces manual errors but also allows you to run many experiments in parallel, compare results systematically, and update your baseline effortlessly.

    Experiment Tracking:
    Invest time early on in setting up a robust experiment tracking system. The clarity and reproducibility this provides will be invaluable as you refine your trading strategy.

    Documentation:
    Keep detailed records of every experiment. Document not just the outcomes, but also the reasoning behind each experiment’s design. This will help guide future improvements and troubleshooting.

By following this roadmap, you’ll have a structured approach to enhancing your forecasting pipeline, automating experiments, and ultimately deploying a well-informed trading system.

Feel free to modify this README as your project evolves. Happy experimenting!

