# Used car regression: auction purchase decisions

This project fits a multiple linear regression to 240 historical retail car sales, tests its predictive accuracy on a held-out 30%, and evaluates a buy/pass decision rule on 40 separate auction lots.

## Run

Install Python 3 and packages:

```bash
pip install pandas numpy statsmodels scikit-learn matplotlib
python analysis.py
```

The script uses `train_test_split(test_size=0.30, random_state=42)` and selects between a full model and a reduced model based solely on training-set adjusted R². The baseline is the training-set average sale price. Neither the held-out test cars nor the 40 auction lots are used to fit or select the model.

The buy rule is `predicted_sale_price > auction_price + 1500`. Actual realized profit on purchased lots is `Later_Sale_Price - Auction_Price - 1500`; passed lots contribute zero.

## Outputs

- `model_summary.txt`: selected OLS coefficients, p-values, and adjusted R²
- `test_metrics.csv`: held-out R² and RMSE for both predictors
- `test_predictions.png`: actual vs predicted test prices
- `auction_decisions.csv`: predictions, buy/pass choices, and potential profits by lot
- `auction_summary.csv`: purchased lots, loss-making purchases, and total profit by strategy

Slides should be authored by the student using the reproduced results and the chart. The script does not generate slides.
