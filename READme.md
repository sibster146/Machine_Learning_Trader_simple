# Machine Learning Paper Trading Simulator
## About
This repo is a paper simulator to test the live performance of machine learning binary classifiers for choosing when to buy and sell stocks, cryptocurrencies, and other tradeable assets. This simulator is connected to the Coinbase exchange, but in theory, can be applied to any exchange and any asset class. You can plug and play any machine learning binary classifier to see its performance. The binary classifier should be trained to predict when the mid price will go up and when it will go down, and the simulator will show how accurately it will be able to do so. NOTE: This simulator should not be used to test PnL performance; the Advanced version on my repo page does this; this simulator only shows how accurately the binary classifier can predict price movement.

## Workflow
### Connecting to Exchange
The simulation starts with reading in exchange data through websocket.py. Each exchange has different websocket APIs; currently, websocket.py is configured to read in data from Coinbase exchange. You have to provide your own API name and secret key to connect to each exchange. After the keys are plugged in, the websocket will continually read in updates and place each update into the `WebSocket.ws_updates_queue`, which is also accessible by `Simulator`.

### Updating Orderbook
After the `WebSocket` places each update on `WebSocket.ws_updates_queue`, the `Simulator` processes each update from `Simulator.websocket_updates_queue` in `Simulator.process_msg(msg)`, where `msg` is an orderbook update. In `Simulator.process_msg(msg)`, the function calls `OrderBook.process_updates(updates)`, which updates `OrderBook`. `OrderBook` contains real-time orderbook data from whichever exchange and asset `WebSocket` is connected to.

### Selling Holdings
After the orderbook is updated, we check if any of our active positions have reached their sell time. All active positions are held in `Simulator.exposure`, which is made up of `Trade`. If 
`Trade.sell_sequence_number <= sequence_number`, then the current `Trade` is ready to sell. If a `Trade` is ready to sell, we pop it from `Simulator.exposure`, update `Simulator.pnl`, and place the `Trade` on `Simulator.completed_trades_queue`. This queue writes to a csv file of all the trades of the simulation.

### Machine Learning Price Movement Prediction
After expired active positions are sold, we check if the current state of the orderbook might lead to an increase or decrease in mid price. We call `Simulator.binary_classifier.create_inference_vector(bids, asks, timestamp_str)`. This function returns a boolean, `1` means price will go up. If `1`, we place an active `Trade` position into `Simulator.exposure`. 

