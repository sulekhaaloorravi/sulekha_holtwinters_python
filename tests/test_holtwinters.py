from sulekha_holtwinters_python import holtwinters as hw
hd = hw.holtwinters()
import pandas as pd
import pkg_resources
DB_FILE = pkg_resources.resource_filename('sulekha_holtwinters_python', 'data')
testDF =pd.read_csv(DB_FILE)
testDF
accuracy1, alpha1, beta1, gamma1, seasonality1 = hd.BestFit(testDF, possible_season_lengths = [12, 24, 36], n_predictions = 48, model_type='multiplicative')
accuracy2, alpha2, beta2, gamma2, seasonality2 = hd.BestFit(testDF, possible_season_lengths = [12, 24, 36], n_predictions = 48, model_type='additive')
Observed1, Predictions1, Level1, Trend1, Seasonality1 = hd.holtwinters(testDF,alpha1, beta1, gamma1, seasonality1,48,'additive')
hd.RMSE(Observed1, Predictions1)
hd.ABSError(Observed1, Predictions1)
hd.MAPE(Observed1, Predictions1)
hd.Accuracy(Observed1, Predictions1)
Observed2, Predictions2, Level2, Trend2, Seasonality2 = hd.holtwinters(testDF,alpha2, beta2, gamma2, seasonality2,48,'multiplicative')
hd.RMSE(Observed2, Predictions2)
hd.ABSError(Observed2, Predictions2)
hd.MAPE(Observed2, Predictions2)
hd.Accuracy(Observed2, Predictions2)
hd.initiate_trend_factor(testDF,24)
SeasonalIndices = hd.initiate_seasonal_indices(testDF,24, model_type = 'multiplicative')
SeasonalIndices2 = hd.initiate_seasonal_indices(testDF,24, model_type = 'additive')
model1 = Observed1, Predictions1, Level1, Trend1, Seasonality1
hd.CreateGraphs(model1)
model2 = Observed2, Predictions2, Level2, Trend2, Seasonality2
hd.CreateGraphs(model2)