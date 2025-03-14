from websocket import WebSocket
from queue import Queue
import threading
from orderbook import OrderBook
from collections import deque
import traceback
from trade import Trade
from datetime import datetime
import os
import csv

class Simulator:
    def __init__(self, symbol, binary_classifier, simulation_run):
        self.pnl = 0
        self.unrefined_pnl = 0
        self.symbol = symbol
        self.binary_classifier = binary_classifier
        self.orderbook = OrderBook()
        self.websocket_updates_queue = Queue() # updates from exchange
        self.completed_trades_queue = Queue() 
        self.price_level_queue = Queue()
        self.exposure = deque() 
        self.websocket = WebSocket(self.websocket_updates_queue)
        self.process_websocket_updates_queue_thread = threading.Thread(target=self.process_websocket_updates_queue)
        self.process_completed_trades_queue_thread = threading.Thread(target = self.process_completed_trades_queue)
        self.process_price_level_thread = threading.Thread(target = self.process_price_level_queue)

        self.completed_trades_filename = f"simulations/{simulation_run}.csv"
        
        self.price_level_filename = f"/Users/sibysuriyan/Documents/code_projects/All_Trading_Applicatoins/price_levels/{simulation_run}.csv"

    def start(self):
        print("starting application")
        self.process_websocket_updates_queue_thread.start()
        self.process_completed_trades_queue_thread.start()
        self.process_price_level_thread.start()
        self.websocket.open_socket(self.symbol)

    def stop(self):
        print("closing application")
        self.websocket.close_socket()

        self.websocket_updates_queue.put(None)
        self.completed_trades_queue.put(None)
        self.price_level_queue.put(None)

        self.process_websocket_updates_queue_thread.join()
        self.process_completed_trades_queue_thread.join()
        self.price_level_queue.join()

    def process_completed_trades_queue(self):
            
        while True:
            completed_trade = self.completed_trades_queue.get()
            if completed_trade == None:
                return
            attributes = completed_trade.__dict__
            if not os.path.exists(self.completed_trades_filename):
                column_names = list(attributes.keys())
                with open(self.completed_trades_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(column_names)

            values = list(attributes.values()) 
            with open(self.completed_trades_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(values)

            self.completed_trades_queue.task_done()


    def process_price_level_queue(self):
        while True:
            data = self.price_level_queue.get()
            if not data:
                return
            bids, asks, timestamp, sequence_number = data
            if not os.path.exists(self.price_level_filename):
                price_level_num = len(bids)
                column_names = []
                column_names.append("Sequence Number")
                for i in range(price_level_num):
                    column_names.append(f"Bid Price Level {i+1}")
                    column_names.append(f"Bid Volume Level {i+1}")
                for i in range(price_level_num):
                    column_names.append(f"Ask Price Level {i+1}")
                    column_names.append(f"Ask Volume Level {i+1}")
                column_names.append("Timestamp")
                with open(self.price_level_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(column_names)

            values = []
            values.append(sequence_number)
            for i in range(len(bids)):
                values.append(bids[i][0])
                values.append(bids[i][1])
            for i in range(len(asks)):
                values.append(asks[i][0])
                values.append(asks[i][1])
            values.append(timestamp)

            with open(self.price_level_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(values)

            self.price_level_queue.task_done()


    def process_msg(self, msg_json):
        try:

            if "updates" in msg_json["events"][0]:

                sequence_number = msg_json["sequence_num"]
                updates = msg_json["events"][0]["updates"]
                self.orderbook.process_updates(updates)
                curr_mid_price = self.orderbook.get_mid_price()

                while self.exposure and (self.exposure[0].sell_sequence_number <= sequence_number): # or self.exposure[0].buy_price < curr_mid_price):
                    sold_trade = self.exposure.popleft()
                    sold_trade_buy_price = sold_trade.buy_price
                    self.pnl += (curr_mid_price - sold_trade_buy_price)
                    sold_trade.update(sell_price = curr_mid_price, pnl = self.pnl)
                    self.completed_trades_queue.put(sold_trade)
                    print(self.pnl)




                bids, asks = self.orderbook.get_n_level_bids_asks(self.binary_classifier.price_level_num)
                timestamp_str = msg_json["timestamp"]
                self.price_level_queue.put((bids,asks,timestamp_str, sequence_number))
                up = self.binary_classifier.create_inference_vector(bids, asks, timestamp_str)

                if up:
                    update_lag = self.binary_classifier.update_lag
                    sell_sequence_num = sequence_number + update_lag
                    refined_trade = Trade(buy_price=curr_mid_price, buy_sequence_number=sequence_number, sell_sequence_number=sell_sequence_num)
                    self.exposure.append(refined_trade)


        except Exception as e:
            print(" Error in processing update ", e, "\n")
            traceback.print_exc()

    

    def process_websocket_updates_queue(self):
        while True:
            msg_json = self.websocket_updates_queue.get()
            if msg_json == None:
                return
            
            self.process_msg(msg_json)
            self.websocket_updates_queue.task_done()

