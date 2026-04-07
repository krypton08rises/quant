def get_data():
    """
    Get trading data from both yfinance and other APIs
    perhaps in the future text(news data...)
    """
    pass


def get_signal(data):
    """
        use some logic of the nbeatsx model(for now) and obv / volatility data
        to generate buy/sell indicators  --> for eg.

    for i in range(len(data)):
        # Buy if momentum is positive and OBV is increasing
        if data['Momentum'][i] > 0 and data['OBV'][i] > data['OBV'][i-1]:
            buy_signals.append(data['close'][i])
            sell_signals.append(np.nan)
        # Sell if momentum is negative
        elif data['Momentum'][i] < 0:
            sell_signals.append(data['close'][i])
            buy_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    return buy_signals, sell_signals

    """

    pass
