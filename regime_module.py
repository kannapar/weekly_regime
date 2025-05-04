import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from yahooquery import Ticker
from datetime import datetime as dt
import cvxpy as cp
from scipy.stats import norm
import config as c

# set variables
start_date = c.StartDate
end_date = c.EndDate
steps_per_year = c.StepsPerYear
data_is_rets= c.ReturnRets
num_regime = c.NumRegime
lambda_value= c.Lambda
normal_regime_val = c.NormalRegimeVal
crash_regime_val = c.CrashRegimeVal

class TickerData:
    """
    outputs stock ticker data
    for given start and end dates
    returns or prices as specified
    with daily/weekly/monthly frequency 
    ***********************************
    HOW TO USE:
    initiate class with ticker,start_date, end_date,data_is_rets, freq
    use get_ticker_data method to get data
    ex: 
    start = dt(1990,1,1)
    end = dt(2024,12,31)
    weekly_data = TickerData('AAPL, MSFT', start, end, True,'weekly')
                        .get_ticker_data()
 
    """
    def __init__(self,ticker):
        self.ticker = ticker
    
    def get_data(self):
        """
        returns close price data of stock given by ticker
        """
        self.print_invalid_ticker()
        yahoo_ticker = Ticker(self.ticker, asynchronous=True)
        ticker_data = yahoo_ticker.history(start=start_date, end=end_date, 
                                           interval = '1d', adj_ohlc= True).reset_index()
        # to get time (hr,min,sec) into date format
        ticker_data['date']= pd.to_datetime(ticker_data['date']).dt.date
        # pivot table 
        daily_prices = ticker_data.pivot(index='date', columns='symbol', values='close')
        daily_prices.index = pd.to_datetime(pd.to_datetime(daily_prices.index).date)
        
        if data_is_rets:
            daily_returns = self.daily_returns(daily_prices)
            if steps_per_year==12:
                #print('Monthly Returns')
                return self.monthly_returns(daily_returns)
            elif steps_per_year == 52:
                #print('Weekly Returns')
                return self.weekly_returns(daily_returns)
            elif steps_per_year == 252:
                #print('Daily Returns')
                return daily_returns
            else:
                raise ValueError('Specify steps_per_year- 252/52/12')
        else:    
            return daily_prices
    
    def print_invalid_ticker(self):
        """
        prints invalid tickers
        check before pulling data
        """
        t = Ticker(self.ticker, validate=True)
        if t.invalid_symbols is not None:
            print("Invalid Tickers entered: ",t.invalid_symbols)
            print()
        
    def daily_returns(self,daily_prices):
        """
        take daily prices and outputs daily returns 
        """
        #log_returns = np.log(1+returns)
        return daily_prices.pct_change(axis=0).dropna()

    def weekly_returns(self,daily_returns):
        """
        takes in daily returns to output weekly returns
        """
        weekly_returns = daily_returns.resample('W').agg(lambda x: (x + 1).prod() - 1)
        weekly_returns.index = weekly_returns.index.to_period("W")
        # get ending date- week ending on Sunday
        weekly_returns.index = weekly_returns.index.astype('str').str.split("/").str[1]
        return weekly_returns

    def monthly_returns(self,daily_returns):
        """
        takes in daily returns to output monthly returns
        """
        monthly_returns = daily_returns.resample('M').agg(lambda x: (x + 1).prod() - 1)
        monthly_returns.index = monthly_returns.index.to_period("M")
        return monthly_returns
            

        
class RegimeIdentification:
    """
    outputs categorical regime (crash/normal)
    based on input prices/returns data
    number of regimes and lambda value
    can also plot the trend
    ***********************************
    HOW TO USE:
    initiate class with data,num_regime,lambda_value,data_is_returns
    use obtain_regime method to get regime
    use plot_filtered_trend method to plot data and trend
    ex:
    r= RegimeIdentification(weekly_sp500_data,2,0.08,True)
    r.obtain_regime()
    r.plot_filtered_trend()
    """
    def get_sp500_regime(self):
        """
        outputs regime array with dates as index
        based on S&P 500 total returns 
        for given date range and frequency
        """
        sp500_rets = TickerData('^SP500TR').get_data()
        regime_data = self.obtain_regime(sp500_rets)
        #print("Proportion of each Regime: ")
        #print(regime_data.value_counts(normalize=True))
        sp500_rets.plot.kde(
            title=f"S&P 500 {self.convert_frequency(steps_per_year)} Returns Density Plot")
        self.plot_filtered_trend(sp500_rets) 
        return regime_data
    
    # For data other than S&P 500
    def obtain_regime(self,data):
        """
        Output regime from data supplied
        num_regime -> number of regimes
        If data_is_returns = True, supplied data is of returns, ex: 5.4
        otherwise prices
        -1 -> crash, 0-contagion, 1  -> normal
        """
        if not data_is_rets:
            data = data.pct_change().dropna()*100
        betas = pd.Series(self.trend_filtering_algorithm(data),
                          data.index)
        if num_regime==2:
            return betas.apply(self.two_regime)
        if num_regime==3:
             return betas.apply(self.three_regime)
            
        
    def plot_filtered_trend(self,data):
        """
        Plots Original data and fitted trend series
        """
        fig, ax = plt.subplots(figsize=(12,7))
        data.plot(ax=ax, label = data.columns, color= 'gray')
        beta = self.trend_filtering_algorithm(data)
        beta = pd.Series(beta, index = data.index)
        beta.plot( ax= ax, label = 'Trend', color= 'midnightblue')
        plt.title("Orginal Data and Filtered Trend", fontsize=20)
        if data_is_rets:
            plt.ylabel(" Returns %")
        else:
            plt.ylabel(" Prices")
        plt.xlabel("Time(Year)")
        plt.legend()
        
    # Trend Filtering and Regime Identification helper Functions
    def first_order_difference_matrix(self,n):
        """
        generates a (n-1 * n) first order difference matrix
        used when input is a timeseries return data
        in trend filtering alogorithm
        """
        D= np.diag([1]*(n-1),1)- np.diag([1]*n)
        return D[:-1,:]

    def second_order_difference_matrix(self,n):
        """
        generates a (n-2 * n) second order difference matrix
        used when input is an index such as S&P 500 index
        in trend filtering alogorithm
        """
        D = np.diag([1]*n)-np.diag([1]*(n-1),1)- np.diag([1]*(n-1),1) +  np.diag([1]*(n-2),2)
        return D[:-2,:]

    def trend_filtering_algorithm(self,data):

        """
        outputs filtered trend series from original data supplied
        for a given value of lambda
        Convex Optimization Problem- find beta/fitted series
        β_hat = argmin β (||x-β||2)^2 + λ||Dβ||1
        where D is a difference matrix
        and λ>0 is a hyper-parameter that balances goals of
        convergence and regime switches
        if input data is returns (data_is_returns= True),
        a first order difference matrix of shape (n-2 * n) is taken
        otherwise second order difference matrix of shape (n-1 * n)
        for timeseries price/index data
        """
        x= np.array(data).reshape(-1)
        n= x.size
        if data_is_rets:
            D = self.first_order_difference_matrix(n)
        else:
            D = self.second_order_difference_matrix(n)

        beta = cp.Variable(n)
        lambd = cp.Parameter(nonneg= True)
        obj = cp.Minimize((cp.norm(x-beta,2))**2 + lambd*cp.norm(D@beta,1))
        prob = cp.Problem(obj)
        #lambd_values = np.logspace(-2, 3, 50)
        lambd.value = lambda_value
        prob.solve(solver=cp.ECOS)
        return beta.value

    def two_regime(self,betas):
        """
        output 1 for normal
        and -1 for crash
        """
        if betas> 0:
            return normal_regime_val
        else:
            return crash_regime_val

    def three_regime(self,betas):
        """
        FIX ME
        output 1 for normal
        0 for transition and
        -1 for crash
        """
        if betas>= 0.02/52:
            return self.normal
        elif betas <= -0.03/52:
            return self.crash
        else:
            return self.transition

    def convert_frequency(self,steps_per_year):
        if steps_per_year ==252:
            return 'daily'
        elif steps_per_year == 52:
            return 'weekly'
        elif steps_per_year == 12:
            return 'monthly'
        elif steps_per_year == 1:
            return 'annual'
        else:
            print('Specify appropriate steps_per_year in config')