# Trader-Sentiment-Analysis

## Objective
Analyze the relationship between market sentiment (Fear vs Greed) and trader performance using local CSV datasets.

## Dataset Description
- `sentiment.csv` or `fear_greed_index.csv`: daily Bitcoin market sentiment with date and classification.
- `trader_data.csv` or `historical_data.csv`: trader-level execution data with fields such as account, symbol, execution price, size, side, closedPnL, and time.

## Files
- `analysis.ipynb`: main analysis notebook
- `README.md`: project overview and run instructions

## How to Run
1. Open the project folder in VS Code.
2. Open `analysis.ipynb`.
3. Select a Python 3 kernel.
4. Run the cells from top to bottom.

The notebook is written to work with either the filenames from the assignment prompt or the files already present in this workspace.

## Summary Insights
- Fear days show stronger average daily PnL in this workspace than Greed days.
- Traders appear to change behavior by sentiment, especially in trade frequency and side mix.
- High-frequency and higher-leverage traders capture more total PnL, but their results are less stable, so risk controls matter.

## Notes
- Nothing is pushed to GitHub.
- No git commits are created.
- The project is fully local.







