"""Reproducible used-car pricing and auction-decision analysis."""
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sales = pd.read_csv(HERE / 'used_car_sales.csv')
auction = pd.read_csv(HERE / 'auction_lots.csv')
assert len(sales) == 240 and len(auction) == 40
train, test = train_test_split(sales, test_size=0.30, random_state=42)
formula = 'Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident) + Num_Owners'
full = smf.ols(formula, data=train).fit()
print('FULL MODEL SUMMARY')
print(full.summary())
print('\nFull model p-values:\n', full.pvalues)
# Num_Owners is not significant in the training fit; compare a reduced version.
reduced = smf.ols('Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident)', data=train).fit()
print('REDUCED ADJUSTED R2:', reduced.rsquared_adj)
# Choose on training-only adjusted R2, with simplicity breaking a tie.
v1 = reduced if reduced.rsquared_adj >= full.rsquared_adj else full
print('SELECTED MODEL:', 'reduced' if v1 is reduced else 'full')
baseline_mean = train['Sale_Price'].mean()
actual = test['Sale_Price']
baseline_predictions = np.full(len(test), baseline_mean)
v1_predictions = v1.predict(test)
metrics = pd.DataFrame([
    {'Strategy':'Baseline', 'Test_R2':r2_score(actual, baseline_predictions), 'Test_RMSE':np.sqrt(mean_squared_error(actual, baseline_predictions))},
    {'Strategy':'v1', 'Test_R2':r2_score(actual, v1_predictions), 'Test_RMSE':np.sqrt(mean_squared_error(actual, v1_predictions))},
])
print('\nTRAIN/TEST:',len(train),len(test))
print('TRAIN MEAN:',baseline_mean)
print('TRAIN ADJUSTED R2:',v1.rsquared_adj)
print('\nTEST METRICS:\n',metrics.to_string(index=False))

def buy_or_pass(predicted_sale_price, auction_price, lot_cost=1500):
    return 'buy' if predicted_sale_price > auction_price + lot_cost else 'pass'

auction['Baseline_Prediction'] = baseline_mean
auction['v1_Prediction'] = v1.predict(auction)
auction['Baseline_Decision'] = [buy_or_pass(p,a) for p,a in zip(auction['Baseline_Prediction'],auction['Auction_Price'])]
auction['v1_Decision'] = [buy_or_pass(p,a) for p,a in zip(auction['v1_Prediction'],auction['Auction_Price'])]
auction['Realized_Profit_If_Bought'] = auction['Later_Sale_Price'] - auction['Auction_Price'] - 1500
summary=[]
for name in ['Baseline','v1']:
    bought = auction.loc[auction[f'{name}_Decision']=='buy']
    summary.append({'Strategy':name,'Cars_Bought':len(bought),'Money_Losing_Cars':int((bought['Realized_Profit_If_Bought']<0).sum()),'Total_Profit':int(bought['Realized_Profit_If_Bought'].sum())})
summary=pd.DataFrame(summary)
print('\nAUCTION OUTCOMES:\n',summary.to_string(index=False))
print('INCREMENTAL PROFIT:',int(summary.loc[1,'Total_Profit']-summary.loc[0,'Total_Profit']))
print('\nCOEFFICIENTS:\n',pd.DataFrame({'Coefficient':v1.params,'P_value':v1.pvalues}).to_string())
metrics.to_csv(HERE/'test_metrics.csv',index=False)
summary.to_csv(HERE/'auction_summary.csv',index=False)
auction.to_csv(HERE/'auction_decisions.csv',index=False)
with open(HERE/'model_summary.txt','w') as f:
    f.write(str(v1.summary()))
fig,ax=plt.subplots(figsize=(7,5))
ax.scatter(actual,v1_predictions,alpha=.7,label='v1 predictions')
lims=[min(actual.min(),v1_predictions.min())-1000,max(actual.max(),v1_predictions.max())+1000]
ax.plot(lims,lims,linestyle='--',color='gray',label='Perfect prediction')
ax.set(xlabel='Actual sale price ($)',ylabel='Predicted sale price ($)',title='Out-of-sample predictions: 72 test cars')
ax.legend()
fig.tight_layout()
fig.savefig(HERE/'test_predictions.png',dpi=200)
plt.close(fig)
