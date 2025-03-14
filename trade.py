class Trade:
    def __init__(self, buy_price, buy_sequence_number, sell_sequence_number):
        self.buy_price = buy_price
        self.sell_price = None
        self.gain = None

        self.buy_sequence_number = buy_sequence_number
        self.sell_sequence_number = sell_sequence_number
        self.pnl = None


    def update(self, sell_price, pnl):
        self.sell_price = sell_price
        self.gain = sell_price > self.buy_price
        self.pnl = pnl
