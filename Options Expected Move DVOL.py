import asyncio
import websockets
import json
import pandas as pd
import datetime as dt
import time
from matplotlib import pyplot as plt
import numpy as np
import math
import statistics as stats


# Get data from Deribit API websocket
async def call_api(msg):
    async with websockets.connect('wss://www.deribit.com/ws/api/v2/') as websocket:
        await websocket.send(msg)
        while websocket.open:
            response = await websocket.recv()
            return response


# Get all data through async loop
def async_loop(api, message):
    temp = asyncio.get_event_loop().run_until_complete(api(message))
    return temp


# Get price data
def retrieve_daily_closes(start_time, end_time, instrument_name, time_setting):
    msg = \
        {
            "jsonrpc": "2.0",
            "id": 833,
            "method": "public/get_tradingview_chart_data",
            "params": {
                "instrument_name": instrument_name,
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "resolution": time_setting
            }
        }
    resp = async_loop(call_api, json.dumps(msg))

    return resp


def volatility_data(start_time, end_time, currency_name):
    msg = \
        {
            "jsonrpc": "2.0",
            "method": "public/get_volatility_index_data",
            "params": {
                "currency": currency_name,
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "resolution": "1D"
            }
        }
    resp = async_loop(call_api, json.dumps(msg))

    return resp


# Convert JSON info from Deribit API into pandas dataframe
def json_to_dataframe(json_resp):
    res = json.loads(json_resp)

    df = pd.DataFrame(res['result'])

    return df


def plot_data(volatility_dates, volatility_closes, currency_name):
    # Dark background bc sexy
    plt.style.use('dark_background')

    # Turn into ax subplot bc easier to work with
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(currency_name + ' DVOL and Price')

    # Create secondary Y-axis
    ax2 = ax.twinx()

    # Plot both
    ax.plot(volatility_dates, volatility_closes, color='skyblue', label='DVOL')
    ax2.plot(volatility_dates, priceData_df.close, color='w', label=(currency_name + ' Price'))

    # set frequency and rotation of dates on x-axis
    ax.set_xticks(np.arange(min(np.arange(len(vol_dates))), max(np.arange(len(vol_dates)))+1, 4))
    fig.autofmt_xdate()

    # give labels
    ax.set_xlabel('Date', color='w')
    ax.set_ylabel('DVOL', color='w')
    ax2.set_ylabel(currency_name + ' Price', color='w')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    # style
    plt.rcParams["font.size"] = 7
    plt.tight_layout()
    plt.rcParams["font.family"] = "monospace"

    # plot
    plt.show()


def plot_price_range(price_df, upper_range, lower_range, upper_range2, lower_range2, currency_name):
    # Convert timestamp into proper datetime
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    x = np.arange(0, len(price_df))

    # Dark background bc sexy
    plt.style.use('dark_background')

    # Make subplot
    fig, ax = plt.subplots(1, figsize=(12, 6))

    # OHLC candles from https://towardsdatascience.com/basics-of-ohlc-charts-with-pythons-matplotlib-56d0e745a5be
    for idx, val in price_df.iterrows():
        if idx < len(price_df)-1:
            # High/low lines
            ax.plot([x[idx], x[idx]], [val['low'], val['high']], color='white')

            # Open Marker
            ax.plot([x[idx], x[idx] - 0.2], [val['open'], val['open']], color='white')

            # Close Marker
            ax.plot([x[idx], x[idx] + 0.2], [val['close'], val['close']], color='white')
        else:
            # Last candle is different color because it is in progress
            ax.plot([x[idx], x[idx]], [val['low'], val['high']], color='green', label='Candle in progress')
            ax.plot([x[idx], x[idx] - 0.2], [val['open'], val['open']], color='green')
            ax.plot([x[idx], x[idx] + 0.2], [val['close'], val['close']], color='green')


    # ticks
    plt.xticks(x[::4], price_df['timestamp'].dt.date[::4])
    plt.xticks(rotation=45)

    # give labels
    ax.set_xlabel('Date', color='w')
    ax.set_ylabel('Price', color='w')
    plt.title(currency_name + ' Price with expected move from DVOL')

    # style
    plt.tight_layout()
    plt.rcParams["font.family"] = "monospace"
    ax.grid(which='major', axis='y', linestyle='--', alpha=0.5, color='w')

    # plot upper and lower bands
    ax.plot(x, upper_range, label='1 SD Upper Range', color='skyblue')
    ax.plot(x, lower_range, label='1 SD Lower Range', color='mediumorchid')
    ax.plot(x, upper_range2, label='2 SD Upper Range', color='mediumblue', alpha=0.75, linestyle='--')
    ax.plot(x, lower_range2, label='2 SD Lower Range', color='darkviolet', alpha=0.75, linestyle='--')

    # legend
    ax.legend()

    # plot
    plt.show()


if __name__ == '__main__':
    for p in range(2):
        currency = ['BTC', 'ETH']

        # Input to receive closing data from currency
        start = int(dt.datetime(2021, 4, 1, 9).timestamp()*1000)
        end = int(time.time()) * 1000
        print(start, end)
        instrument = ["BTC-PERPETUAL", 'ETH-PERPETUAL']
        timeframe = '1D'

        # Receive closing data in JSON
        json_daily_closes = retrieve_daily_closes(start, end, instrument[p], timeframe)

        # Convert JSON data into pandas dataframe
        priceData_df = json_to_dataframe(json_daily_closes)

        # Convert unix into date
        priceData_df['ticks'] = priceData_df.ticks / 1000
        priceData_df['timestamp'] = [dt.datetime.fromtimestamp(date).strftime('%Y-%m-%d') for date in priceData_df.ticks]

        # Get DVOL data from API in JSON
        json_vol_data = volatility_data(start, end, currency[p])

        # Convert JSON data to pandas dataframe
        volatility_df = json_to_dataframe(json_vol_data)

        # Get dates and closes of DVOL index and put into list
        vol_dates = []
        vol_closes = []
        for i in range(len(volatility_df.data)):
            vol_dates.append(dt.datetime.utcfromtimestamp((volatility_df.data[i][0] / 1000) + 86400).strftime('%m-%d'))
            vol_closes.append(volatility_df.data[i][3])

        # Plot DVOL and BTC price
        #plot_data(vol_dates, vol_closes)
        print(priceData_df.head())

        # Initialise variables
        expMove = []
        exp_move_up = []
        exp_move_down = []
        exp_move_up2 = []
        exp_move_down2 = []
        lower_move_diff = []
        upper_move_diff = []

        # Calculate expected move, expected ranges and basic stat data
        for j in range(len(vol_closes)-1):
            # Expected Move Calculation: Volatility / Sqrt(365) * Price
            expMove = ((vol_closes[j+1]/100)/math.sqrt(365))*priceData_df.close[j]

            # Price +- Expected move to get range
            exp_move_up.append(priceData_df.close[j] + expMove)
            exp_move_down.append(priceData_df.close[j] - expMove)

            # 2SD range
            exp_move_up2.append(priceData_df.close[j] + (expMove*2))
            exp_move_down2.append(priceData_df.close[j] - (expMove*2))

            # Print Data
            print('Date: ', vol_dates[j], 'Open Price: ', priceData_df.close[j], ', DVOL: ', vol_closes[j+1],
                  ', Expected move: ', round(expMove, 1),  ', 68% CI Upper-range: ', round(exp_move_up[j], 1), ', 68% CI Lower-range: ',
                  round(exp_move_down[j], 1),'| 95% Upper-range: ', round(exp_move_up2[j], 1), ', 95% Lower-range: ', round(exp_move_down2[j],1))

            # Get Difference between actual lows and high and expected range
            lower_move_diff.append(abs(priceData_df.low[j] - exp_move_down[j]) / exp_move_down[j])
            upper_move_diff.append(abs(priceData_df.high[j] - exp_move_up[j]) / exp_move_up[j])

        # Print Data
        print('On Average, the low is ', str(round(stats.mean(lower_move_diff) * 100, 2)) +
              '% off (higher or lower than) the expected low, and the median difference is ',
              str(round(stats.median(lower_move_diff) * 100, 2)) + '%')
        print('On Average, the high is ', str(round(stats.mean(upper_move_diff) * 100, 2)) +
              '% off (higher or lower than) the expected high, and the median difference is ',
              str(round(stats.median(upper_move_diff) * 100, 2)) + '%')

        # Insert one more so plot works bit of a bandaid fix
        exp_move_up.insert(0, exp_move_up[0])
        exp_move_down.insert(0, exp_move_down[0])
        exp_move_up2.insert(0, exp_move_up2[0])
        exp_move_down2.insert(0, exp_move_down2[0])

        plot_price_range(priceData_df, exp_move_up, exp_move_down, exp_move_up2, exp_move_down2, currency[p])


