import asyncio
import itertools
import websockets
import json
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
import concurrent.futures
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import humanize

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


# async loop for threading so new async loop for each thread
def async_loop_thr(api, message):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    temp = loop.run_until_complete(api(message))
    loop.close()
    return temp


# get current btc price
def get_currency_price(crypto):
    msg = \
        {
            "method": "public/get_index_price",
            "params": {
                "index_name": crypto
            },
            "jsonrpc": "2.0",
            "id": 4
        }

    resp = async_loop(call_api, json.dumps(msg))
    return resp


# get names of all options on book
def retrieve_options_names(crypto):
    msg = \
        {
            "jsonrpc": "2.0",
            "id": 7617,
            "method": "public/get_instruments",
            "params": {
                "currency": crypto,
                "kind": "option",
                "expired": False
            }
        }

    resp = async_loop(call_api, json.dumps(msg))
    return resp


# Convert JSON info from Deribit API into pandas dataframe
def json_to_dataframe(json_resp):
    res = json.loads(json_resp)

    df = pd.DataFrame(res['result'])

    return df


def retrieve_options_data(instrument_name):
    msg = \
        {
            "jsonrpc": "2.0",
            "method": "public/get_order_book",
            "params": {
                "instrument_name": instrument_name,
                # "depth": 5
            }
        }
    resp = async_loop_thr(call_api, json.dumps(msg))
    return resp


# get data of options including gamma, oi, etc
def options_data_create(name_of_option):
    options_info = []
    json_options_data = json.loads(retrieve_options_data(name_of_option))

    # split up instrument name so that strike price can be isolated
    temp_split = json_options_data['result']['instrument_name'].split('-')

    # Get K = -1 if put or 1 if call
    if json_options_data['result']['instrument_name'][-1] == 'P':
        type = -1
    else:
        type = 1

    # Create list of all relevant data to turn into dataframe later (name, gamma, oi, P/C, strike)
    options_info.append([json_options_data['result']['instrument_name'],
                         json_options_data['result']['greeks']['gamma'],
                         json_options_data['result']['open_interest'],
                         type, int(temp_split[2])])
    print(options_info)
    return options_info


def dataframe_create(options_names):
    futures_list = []
    results = []
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        for p in range(len(options_names)):
        # for p in range(10):
            futures = executor.submit(options_data_create, options_names[p])
            # print(wait(futures_list))
            futures_list.append(futures)
        for future in futures_list:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception:
                results.append(None)
    return results


def plot_gex(price, gex, oi, flip, gamma_total):
    # Dark background bc sexy
    plt.style.use('dark_background')

    # Turn into ax subplot bc easier to work with
    fig, ax = plt.subplots(1, figsize=(12, 6))

    # colour
    cmap = mpl.colors.LinearSegmentedColormap.from_list('white_to_blue', ['lavender', 'mediumblue'])
    norm = plt.Normalize(min(oi), max(oi))
    colors = cmap(norm(oi))
    ax.bar(price, gex, color=colors, edgecolor='white', linewidth=1, width=0.4)

    cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), location='bottom', anchor=(0.95, 1.2), shrink=0.15)
    cbar.ax.tick_params(size=0)
    cbar.set_ticks([])
    # cbar.set_ticklabels([0, 1])
    cbar.set_label('Open Interest', fontsize='x-small')

    # give labels
    plt.rcParams["font.family"] = "monospace"
    ax.set_xlabel('Strike Price (USD)', color='w', fontsize='large', labelpad=5)
    ax.set_ylabel("Total GEX (BTC per $100 move)", color='w')
    plt.title('BTC GEX at strike - ' + dt.datetime.now().strftime('%d/%m/%y'))

    # style
    plt.tight_layout()
    # ax.grid(which='major', axis='x', linestyle='--', alpha=0.5, color='w')
    ax.axhline(0, color='white')
    plt.setp(ax.get_xticklabels()[::2], visible=False)
    plt.xticks(rotation=45)
    # plt.axvline(str(flip), color="w", linestyle="--", lw=2, label="Gamma Flip Price")

    # ax.legend(loc='best')
    plt.show()


def bootleg_flip_calc(strikes):
    print(strikes)

    def _aux_add(a, b):
        print(b[0], a[1] + b[1])
        return b[0], a[1] + b[1]

    cumsum = list(itertools.accumulate(strikes, _aux_add))
    print(cumsum)
    if cumsum[len(strikes) // 10][1] < 0:
        op = min
    else:
        op = max
    print(op)
    return op(cumsum, key=lambda i: i[1])[0]


if __name__ == '__main__':
    # currency type (bitcoin or eth)
    currency = 'BTC'

    # crypto ticker
    crypto_ticker = 'btc_usd'

    # get crypto price
    currency_price_json = json.loads(get_currency_price(crypto_ticker))
    currency_price = currency_price_json['result']['index_price']

    # get all option names
    json_options = retrieve_options_names(currency)
    option_names_df = json_to_dataframe(json_options)
    option_names = option_names_df['instrument_name'].tolist()
    json_results = dataframe_create(option_names)

    fixed_results = []
    print(json_results)
    for i in range(len(json_results)):
        fixed_results.append(json_results[i][0])

    # organise dataframe
    options_data_df = pd.DataFrame(fixed_results, columns=['Name', 'Gamma', 'OI', 'PutCall', 'Price'])
    options_data_df = options_data_df.sort_values(by=['Price'], ascending=True, ignore_index=True)

    # calculate gex with equation OI * gamma * 1/-1 (*100 for gex per $100 move)
    options_data_df['gex'] = options_data_df['OI'] * options_data_df['Gamma'] * options_data_df['PutCall'] * 100

    # sum gamma and OI for same strike
    sorted_data_df = options_data_df.groupby(['Price'], as_index=False)['gex'].sum()
    oi_data_df = options_data_df.groupby(['Price'], as_index=False)['OI'].sum()

    # get irrelevant gamma levels
    irrelevant_gex = []
    min_price = (currency_price // 2)
    max_price = currency_price + (currency_price // 2)
    median_gex = sorted_data_df['gex'].abs().median()

    # see if there is gamma larger than the median outside the default range of +-50%. If there is, extend
    # min/max price to that so that the graph can include these gamma levels as they could be relevant (speculation)
    for f in range(len(sorted_data_df)):
        if (abs(sorted_data_df['gex'][f]) > median_gex) & (sorted_data_df['Price'][f] < min_price):
            min_price = sorted_data_df['Price'][f]
        if (abs(sorted_data_df['gex'][f]) > median_gex) & (sorted_data_df['Price'][f] > max_price):
            max_price = sorted_data_df['Price'][f]

    # remove irrelevant gamma levels
    for irrel in range(len(sorted_data_df)):
        if (sorted_data_df['Price'][irrel] < min_price) | (sorted_data_df['Price'][irrel] > max_price):
            irrelevant_gex.append(irrel)

    options_relevant_df = sorted_data_df.drop(labels=irrelevant_gex, axis=0)
    oi_data_df = oi_data_df.drop(labels=irrelevant_gex, axis=0)

    # to list because easier to work with in plot
    strike_price = options_relevant_df['Price'].tolist()
    total_gex = options_relevant_df['gex'].tolist()
    oi_data = oi_data_df['OI'].tolist()
    flip_calc = []

    # create a tuple of strike and gamma as it is easier for bootleg flip calc
    for a, b in zip(sorted_data_df.Price, sorted_data_df.gex):
        strikes_gamma = (a, b)
        flip_calc.append(strikes_gamma)
    sorted_data_df['StrikeAndGamma'] = flip_calc

    bootleg_flip = bootleg_flip_calc(sorted_data_df.StrikeAndGamma)
    print(bootleg_flip)

    # convert to string for bar graph
    for strcon in range(len(strike_price)):
        strike_price[strcon] = str(strike_price[strcon])

    total_gamma = humanize.intword((sorted_data_df['gex'].sum()) * currency_price)

    print(strike_price, total_gex, oi_data, bootleg_flip)

    plot_gex(strike_price, total_gex, oi_data, bootleg_flip, total_gamma)


