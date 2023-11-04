import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import meanabs, mse, rmse, rmspe


def predict_timeseries(training_data, ar=0, i=0, ma=1, steps=8, **kwargs):
    arima_model = ARIMA(training_data, order=(ar, i, ma))
    model = arima_model.fit()
    predictions = model.simulate(steps, anchor='end')
    return predictions


def calculate_statistics(y_test, predictions):
    stats = dict()
    for name, function in [('Mean Absolute Error', meanabs), ('Mean Squared Error', mse),
                           ('Root Mean Squared Error', rmse), ('Root Mean Squared Percentage Error', rmspe)]:
        stats[name] = function(y_test, predictions)
    table = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    return table


if __name__ == "__main__":
    # Code for debugging/testing
    input = np.arange(0.0, 8.0, 0.15)
    data = 2*np.sin(input*2)
    result = predict_timeseries(data, 2, 0, 1)
    print(print(result['epochs'], result['loss']))
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lineplot(x=range(len(data)), y=data, color='r')
    sns.lineplot(x=range(30, 30+len(result['result'])), y=result['result'], color='b')
    plt.show()
