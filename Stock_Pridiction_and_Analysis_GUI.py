import matplotlib.pyplot as plt
import math
import requests
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import webbrowser
from datetime import date, timedelta
import yfinance as yf
import tkinter as tk
import numpy as np
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Stock:
    def __init__(self, root):
        self.root = root
        #self.root.geometry("1530x790+0+0")
        self.root.attributes('-fullscreen', True)
        self.root.title("Stock Analysis")

        self.var_Symbol=tk.StringVar()
        self.var_start_date = tk.StringVar()
        self.var_MA = tk.StringVar()
        self.var_Days= tk.StringVar()
        self.var_Epochs = tk.StringVar()

        self.var_Language=tk.StringVar()
        self.var_SortBy = tk.StringVar()
        self.var_Query = tk.StringVar()
        self.var_No_Articles = tk.StringVar()
        self.var_API = tk.StringVar()

        self.bg = ImageTk.PhotoImage(file="C:/Users/Utkarsh/Desktop/SIP/SIP_Attendance/Images/img1.jpg")
        label_bg = tk.Label(self.root, image=self.bg)
        label_bg.place(x=0, y=0, relwidth=1, relheight=1)

        title = tk.Label(self.root, text="Stock Analysis", font=("liquid crystal", 30), bg="darkred", fg="white")
        title.place(x=0, y=0, relwidth=1)

        Left_frame = tk.LabelFrame(self.root, bd=2, bg="white", relief=tk.RIDGE, text="Stock Analysis",font=("liquid crystal", 12, "bold"))
        Left_frame.place(x=6, y=590, width=760, height=240)

        RMSE_Label = tk.LabelFrame(self.root, bd=2, bg="white", relief=tk.RIDGE, text="RMSE",
                                font=("liquid crystal", 12, "bold"))
        RMSE_Label.place(x=565, y=610, width=150, height=80)

        Search_news = tk.LabelFrame(self.root, bd=2, bg="white", relief=tk.RIDGE, text="News",font=("liquid crystal", 12, "bold"))
        Search_news.place(x=780, y=590, width=740, height=240)


        self.Graph_frame = tk.LabelFrame(self.root, bd=2, bg="white", relief=tk.RIDGE, text="Graph",font=("liquid crystal", 12, "bold"))
        self.Graph_frame.place(x=75, y=60, width=1390, height=520)

        # Symbol
        Symbol = tk.Label(Left_frame, text="Symbol : ", font=("liquid crystal", 12, "bold"))
        Symbol.grid(row=0, column=0, padx=10, pady=5)

        self.Symbol_box = ttk.Entry(Left_frame, textvariable=self.var_Symbol, width=25, font=("liquid crystal", 12, "bold"))
        self.Symbol_box.grid(row=0, column=1, padx=10, pady=5)
        self.Symbol_box.bind("<KeyRelease>", self.update_ticker)  # Binds KeyRelease event

        #Start Date
        start_date = tk.Label(Left_frame, text="Start date : ", font=("liquid crystal", 12, "bold"))
        start_date.grid(row=1, column=0, padx=10, pady=5)

        self.date_box = ttk.Entry(Left_frame, textvariable=self.var_start_date, width=25, font=("liquid crystal", 12, "bold"))
        self.date_box.grid(row=1, column=1, padx=10, pady=5)
        self.date_box.bind("<KeyRelease>", self.update_StartDate)

        # Moving Average
        Moving_Avg = tk.Label(Left_frame, text="Moving Average : ", font=("liquid crystal", 12, "bold"))
        Moving_Avg.grid(row=3, column=0, padx=10, pady=5)

        self.Moving_Avg_box = ttk.Entry(Left_frame, textvariable=self.var_MA, width=25,font=("liquid crystal", 12, "bold"))
        self.Moving_Avg_box.grid(row=3, column=1, padx=10, pady=5)
        self.Moving_Avg_box.bind("<KeyRelease>", self.MA)

        # Epochs
        Epochs = tk.Label(Left_frame, text="Epochs : ", font=("liquid crystal", 12, "bold"))
        Epochs.grid(row=5, column=0, padx=10, pady=5)

        self.Epochs_box = ttk.Entry(Left_frame, textvariable=self.var_Epochs, width=25,font=("liquid crystal", 12, "bold"))
        self.Epochs_box.grid(row=5, column=1, padx=10, pady=5)
        self.Epochs_box.bind("<KeyRelease>", self.Epoch)

        # Train Data on how many days

        Days = tk.Label(Left_frame, text="Past how many days to Train data : ", font=("liquid crystal", 12, "bold"))
        Days.grid(row=4, column=0, padx=10, pady=5)

        self.Days_box = ttk.Entry(Left_frame, textvariable=self.var_Days, width=25,font=("liquid crystal", 12, "bold"))
        self.Days_box.grid(row=4, column=1, padx=10, pady=5)
        self.Days_box.bind("<KeyRelease>", self.Train_days)

        Exit = tk.Button(self.root, command=self.exit, text="Exit", font=("liquid crystal", 15, "bold"), bd=3,relief=tk.RIDGE, fg="white", bg="red", activebackground="red", activeforeground="white")
        Exit.place(x=0, y=0, width=120, height=35)

        Predict = tk.Button(Left_frame, command=self.predict, text="Predict", font=("liquid crystal", 15, "bold"), bd=3,relief=tk.RIDGE, fg="white", bg="red", activebackground="red", activeforeground="white")
        Predict.place(x=10, y=175, width=120, height=35)

        MA = tk.Button(Left_frame, command=self.Moving_Average, text="MA", font=("liquid crystal", 15, "bold"), bd=3,relief=tk.RIDGE, fg="white", bg="red", activebackground="red", activeforeground="white")
        MA.place(x=140, y=175, width=120, height=35)

        RSI = tk.Button(Left_frame, command=self.RSI, text="RSI", font=("liquid crystal", 15, "bold"), bd=3,relief=tk.RIDGE, fg="white", bg="red", activebackground="red", activeforeground="white")
        RSI.place(x=270, y=175, width=120, height=35)

        Support_resistance = tk.Button(Left_frame, command=self.identify_support_resistance, text="Support/Resistence levels", font=("liquid crystal", 15, "bold"), bd=3,relief=tk.RIDGE, fg="white", bg="red", activebackground="red", activeforeground="white")
        Support_resistance.place(x=400, y=175, width=300, height=35)

        Buy_Sell = tk.Button(Left_frame, command=self.buy_sell, text="Buy/Sell", font=("liquid crystal", 15, "bold"), bd=3,relief=tk.RIDGE, fg="white", bg="blue", activebackground="red", activeforeground="white")
        Buy_Sell.place(x=550, y=105, width=180, height=60)

        #Serach 

        #SortBy
        SortBy = tk.Label(Search_news, text="Sort By : ", font=("liquid crystal", 16, "bold"))
        SortBy.grid(row=0, column=0, padx=10, pady=5)

        # Define sort options
        sort_options = ["Date", "Relevance", "Popularity", "Author", "Source", "Rating", "Trending", "Custom"]

        # Create a Combobox for sorting options
        self.SortBy_box = ttk.Combobox(Search_news, textvariable=self.var_SortBy, values=sort_options, width=25, font=("liquid crystal", 16, "bold"))
        self.SortBy_box.grid(row=0, column=1, padx=10, pady=5)
        self.SortBy_box.bind("<<ComboboxSelected>>", self.temp_SortBy) 

        # Query
        Query = tk.Label(Search_news, text="Query : ", font=("liquid crystal", 16, "bold"))
        Query.grid(row=1, column=0, padx=10, pady=5)

        self.Query_box = ttk.Entry(Search_news, textvariable=self.var_Query, width=25,font=("liquid crystal", 16, "bold"))
        self.Query_box.grid(row=1, column=1, padx=10, pady=5)
        self.Query_box.bind("<KeyRelease>", self.temp_Query)

        #No. of articles
        No_Articles = tk.Label(Search_news, text="Number of Articles  : ", font=("liquid crystal", 16, "bold"))
        No_Articles.grid(row=2, column=0, padx=10, pady=5)

        self.No_Articles_box = ttk.Entry(Search_news, textvariable=self.var_No_Articles , width=25,font=("liquid crystal", 16, "bold"))
        self.No_Articles_box.grid(row=2, column=1, padx=10, pady=5)
        self.No_Articles_box.bind("<KeyRelease>", self.temp_No_Articles)

        #No. of articles
        Api_key = tk.Label(Search_news, text="Api key : ", font=("liquid crystal", 16, "bold"))
        Api_key.grid(row=3, column=0, padx=10, pady=5)

        self.Api_key_box = ttk.Entry(Search_news, textvariable=self.var_API , width=25,font=("liquid crystal", 16, "bold"))
        self.Api_key_box.grid(row=3, column=1, padx=10, pady=5)
        self.Api_key_box.bind("<KeyRelease>", self.temp_Api_key)


        Search = tk.Button(Search_news, command=self.news, text="Search", font=("liquid crystal", 14, "bold"), bd=3,relief=tk.RIDGE, fg="white", bg="blue", activebackground="red", activeforeground="white")
        Search.place(x=10, y=160, width=180, height=50)
        

        self.fig = plt.Figure(figsize=(16, 8)) 
        self.ax = self.fig.add_subplot(111) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.Graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    #Functions to add values to the major functions 
    def update_ticker(self, event=None):
        self.ticker = self.var_Symbol.get()  # Updates ticker 

    def update_StartDate(self, event=None):
        self.ddate = self.var_start_date.get() # Updates Start Date 

    def MA(self, event=None):
        self.var_MovA = self.var_MA.get() # Updates MA 

    def Train_days(self, event=None):
        self.var_DDays = self.var_Days.get() # Updates training days

    def Epoch(self, event=None):
        self.var_E = self.var_Epochs.get() # Updates Epoch of LSTN model

    def temp_SortBy(self, event=None):
        self.var_SortBy1 = self.var_SortBy
    def temp_Query(self, event=None):
        self.var_Query1 = self.var_Query

    def temp_No_Articles(self, event=None):
        self.var_No_Articles1 = self.var_No_Articles

    def temp_Api_key(self, event=None):
        self.var_Api_key1 = self.var_API


    def predict(self):
        if self.var_Symbol.get()=="" or self.var_start_date.get()=="" or self.var_Days.get()=="":
            messagebox.showerror("Error","Not all field filled",parent=self.root)
        else:
            try:
                days = int(self.var_DDays)
                today = date.today()
                d1 = today.strftime("%Y-%m-%d")
                end_date = d1
                #d2 = date.today() - timedelta(days=int(self.ddate))   # start date
                #d2 = d2.strftime("%Y-%m-%d")
                start_date = self.ddate

                ticker = self.ticker  # Useing the updated ticker

                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                data = df.filter(["Close"])
                dataset = data.values
                training_data = math.ceil(len(dataset) * 0.8)  # Number of rows
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)

                # Training
                train_data = scaled_data[0:training_data, :]
                x_train = []
                y_train = []

                # days
                for i in range(days, len(train_data)):
                    x_train.append(train_data[i - days:i, 0])
                    y_train.append(train_data[i, 0])  # It will contain the 61st value, which our xdataset should predict
                x_train = np.array(x_train)  # Convert list to numpy array
                y_train = np.array(y_train)  # Convert list to numpy array

                # Reshape x_train for LSTM
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                model = Sequential()
                model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(100, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                model.fit(x_train, y_train, batch_size=1, epochs=int(self.var_E))

                # Test
                test_data = scaled_data[training_data - days:, :]
                x_test = []
                y_test = dataset[training_data:, :]
                for i in range(days, len(test_data)):
                    x_test.append(test_data[i - days:i, 0])

                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                # Predicted price values
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)


                # RMSE
                RMSE_Label = tk.LabelFrame(self.root, bd=2, bg="white", relief=tk.RIDGE, text="RMSE",font=("liquid crystal", 12, "bold"))
                RMSE_Label.place(x=565, y=610, width=180, height=80)
        
                RMSE = np.sqrt(np.mean(predictions - y_test) ** 2)

                R = tk.Label(RMSE_Label, text=f"{RMSE}", font=("liquid crystal", 12, "bold"))
                R.grid(row=0, column=0, padx=10, pady=5)

                train = data[:training_data]
                valid = data[training_data:]
                valid = valid.copy()
                valid['Predictions'] = predictions



                # Clear the previous plot
                self.ax.clear()

                # Visualize the data
                self.ax.set_title('Model')
                self.ax.set_xlabel('Date', fontsize=18)
                self.ax.set_ylabel('Close Price USD', fontsize=18)
                self.ax.plot(train.index, train['Close'])
                self.ax.plot(valid.index, valid[['Close', 'Predictions']])
                self.ax.legend(['Train', 'Val', 'Predictions'], loc='lower right')

                # Draw the plot
                self.canvas.draw()
            except Exception as es:
                messagebox.showerror("Error",f"Due To :{str(es)}",parent=self.root)



    def Moving_Average(self):
        if self.var_Symbol.get()=="" or self.var_start_date.get()=="" or self.var_MA.get()=="" or self.var_Days.get()=="":
            messagebox.showerror("Error","Not all field filled",parent=self.root)
        else:
            try:
                days = int(self.var_DDays)
                today = date.today()
                d1 = today.strftime("%Y-%m-%d")
                end_date = d1
                #d2 = date.today() - timedelta(days=int(self.ddate))   # start date
                #d2 = d2.strftime("%Y-%m-%d")
                start_date = self.ddate

                ticker = self.ticker  # Useing the updated ticker

                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                data = df.filter(["Close"])
                dataset = data.values
                training_data = math.ceil(len(dataset) * 0.8)  # Number of rows
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)

                # Training
                train_data = scaled_data[0:training_data, :]
                x_train = []
                y_train = []

                # days
                for i in range(days, len(train_data)):
                    x_train.append(train_data[i - days:i, 0])
                    y_train.append(train_data[i, 0])  # It will contain the 61st value, which our xdataset should predict
                x_train = np.array(x_train)  # Convert list to numpy array
                y_train = np.array(y_train)  # Convert list to numpy array

                # Reshape x_train for LSTM
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                model = Sequential()
                model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(100, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                model.fit(x_train, y_train, batch_size=1, epochs=int(self.var_E))

                # Test
                test_data = scaled_data[training_data - days:, :]
                x_test = []
                y_test = dataset[training_data:, :]
                for i in range(days, len(test_data)):
                    x_test.append(test_data[i - days:i, 0])

                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                # Predicted price values
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)


                train = data[:training_data]
                valid = data[training_data:]
                valid = valid.copy()
                valid['Predictions'] = predictions

                # Clears the previous plot
                self.ax.clear()
        
                ma = df.Close.rolling(int(self.var_MovA)).mean()


                self.ax.set_title('Model')
                self.ax.set_xlabel('Date', fontsize=18)
                self.ax.set_ylabel('Close Price USD', fontsize=18)
                self.ax.plot(ma, 'black')
                self.ax.plot(train.index, train['Close'])
                self.ax.plot(valid.index, valid[['Close', 'Predictions']])
                self.ax.legend(['Moving Average'f"{self.var_MovA}",'Train', 'Val', 'Predictions'], loc='lower right')

                # Draw the plot
                self.canvas.draw()
            except Exception as es:
                messagebox.showerror("Error",f"Due To :{str(es)}",parent=self.root)

    def RSI(self):
        if self.var_Symbol.get()=="" or self.var_start_date.get()=="":
            messagebox.showerror("Error","Not all field filled",parent=self.root)
        else:
            try:
                today = date.today()
                d1 = today.strftime("%Y-%m-%d")
                end_date = d1
                #d2 = date.today() - timedelta(days=int(self.ddate))   # start date
                #d2 = d2.strftime("%Y-%m-%d")
                start_date = self.ddate

                ticker = self.ticker  # Useing the updated ticker

                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                def calculate_rsi(data, window=14):
                    delta = data.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi

                df['RSI'] = calculate_rsi(df["Close"])
                fig, axs = plt.subplots(2,1,gridspec_kw={"height_ratios":[3,1]},figsize=(15,11))

                self.ax.clear()  # Clear the previous plot
                self.ax.set_title('RSI')
                self.ax.plot(df['RSI'], color="grey", label='RSI')
                self.ax.axhline(y=70, color="r", linestyle="-")  # 70% limit Sell Signal
                self.ax.axhline(y=30, color="g", linestyle="-")  # 30% limit Buy Signal
                self.ax.legend()

                self.canvas.draw()

            except Exception as es:
                messagebox.showerror("Error",f"Due To :{str(es)}",parent=self.root)

    def identify_support_resistance(self):
        if self.var_Symbol.get()=="" or self.var_start_date.get()=="":
            messagebox.showerror("Error","Not all field filled",parent=self.root)
        else:
            try:
                window=20
                sensitivity=2
                today = date.today()
                d1 = today.strftime("%Y-%m-%d")
                end_date = d1
                #d2 = date.today() - timedelta(days=int(self.ddate))   # start date
                #d2 = d2.strftime("%Y-%m-%d")
                start_date = self.ddate

                ticker = self.ticker  # Use the updated ticker

                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                close_prices = df['Close']
                
                # Calculating rolling minimum and maximum
                rolling_min = close_prices.rolling(window=window, min_periods=1).min()
                rolling_max = close_prices.rolling(window=window, min_periods=1).max()
                
                # Calculate support and resistance levels
                support = rolling_min - (rolling_min * sensitivity / 100)
                resistance = rolling_max + (rolling_max * sensitivity / 100)
                
                # Filter out noise
                support = support.dropna()
                resistance = resistance.dropna()
                
                # Identify support levels
                support_levels = []
                for i in range(len(support)):
                    if i == 0:
                        support_levels.append(support.iloc[i])
                    elif support.iloc[i] != support.iloc[i-1]:
                        support_levels.append(support.iloc[i])
                
                # Identify resistance levels
                resistance_levels = []
                for i in range(len(resistance)):
                    if i == 0:
                        resistance_levels.append(resistance.iloc[i])
                    elif resistance.iloc[i] != resistance.iloc[i-1]:
                        resistance_levels.append(resistance.iloc[i])

                self.ax.clear()  # Clear the previous plot
                self.ax.set_title('Support and Resistance Levels')

                self.ax.plot(df.index, df['Close'], color='blue', label='Close Price')
                self.ax.scatter(df.index, df['Close'], color='blue', s=10, label='_nolegend_')
                self.ax.set_xlabel('Date', fontsize=18)
                self.ax.set_ylabel('Price', fontsize=18)
                self.ax.grid(True)
                self.ax.legend()

                # Plot support levels
                for support_level in support_levels:
                    self.ax.axhline(y=support_level, color='green', linestyle='--', linewidth=1, label='Support')

                # Plot resistance levels
                for resistance_level in resistance_levels:
                    self.ax.axhline(y=resistance_level, color='red', linestyle='--', linewidth=1, label='Resistance')

                
                self.canvas.draw()
            except Exception as es:
                messagebox.showerror("Error",f"Due To :{str(es)}",parent=self.root)
    
    def buy_sell(self):
        if self.var_Symbol.get()=="" or self.var_start_date.get()=="" or self.var_MA.get()=="" or self.var_Days.get()=="":
            messagebox.showerror("Error","Not all field filled",parent=self.root)
        else:
            try:
                days = int(self.var_Days.get())  # Corrected from self.var_DDays
                today = date.today()
                d1 = today.strftime("%Y-%m-%d")
                end_date = d1
                start_date = self.ddate

                ticker = self.ticker  # Use the updated ticker

                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                window=20
                sensitivity=2
                close_prices = df['Close']
                
                # Calculating rolling min and max
                rolling_min = close_prices.rolling(window=window, min_periods=1).min()
                rolling_max = close_prices.rolling(window=window, min_periods=1).max()
                
                # Calculate support and resistance levels
                support = rolling_min - (rolling_min * sensitivity / 100)
                resistance = rolling_max + (rolling_max * sensitivity / 100)
                
                # Filter out noise
                support = support.dropna()
                resistance = resistance.dropna()
                
                # Identify support levels
                support_levels = []
                for i in range(len(support)):
                    if i == 0:
                        support_levels.append(support.iloc[i])
                    elif support.iloc[i] != support.iloc[i-1]:
                        support_levels.append(support.iloc[i])
                
                # Identify resistance levels
                resistance_levels = []
                for i in range(len(resistance)):
                    if i == 0:
                        resistance_levels.append(resistance.iloc[i])
                    elif resistance.iloc[i] != resistance.iloc[i-1]:
                        resistance_levels.append(resistance.iloc[i])

                ma = df.Close.rolling(int(self.var_MA.get())).mean()

                # Buy or sell decision based on the relationship between close price and moving average
                buy_points = []
                sell_points = []
                for i in range(1, len(df)):
                    if df['Close'].iloc[i] > ma.iloc[i] and df['Close'].iloc[i-1] <= ma.iloc[i-1]:
                        buy_points.append((df.index[i], df['Close'].iloc[i]))
                    elif df['Close'].iloc[i] < ma.iloc[i] and df['Close'].iloc[i-1] >= ma.iloc[i-1]:
                        sell_points.append((df.index[i], df['Close'].iloc[i]))

                self.ax.clear()  
                
                # Plotting stock's close price
                self.ax.plot(df.index, df['Close'], label='Close Price')
                
                # Plotting moving average
                self.ax.plot(ma.index, ma, label='Moving Average', color='red')

                # Buy points
                for point in buy_points:
                    self.ax.scatter(point[0], point[1], color='green', marker='^', label='Buy Point', s=100)
                    print(point[0])

                #Sell points
                for point in sell_points:
                    self.ax.scatter(point[0], point[1], color='red', marker='v', label='Sell Point', s=100)

                self.ax.set_title('Buy and Sell Points')
                self.ax.set_xlabel('Date', fontsize=18)
                self.ax.set_ylabel('Close Price', fontsize=18)
                self.ax.grid(True)
                self.canvas.draw()

                self.display_window = tk.Toplevel(self.root)
                self.display_window.title("Buy/Sell Points")
                self.display_window.geometry("400x300")
                
                #listbox to display buy and sell points
                self.points_listbox = tk.Listbox(self.display_window)
                self.points_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar = tk.Scrollbar(self.display_window, orient=tk.VERTICAL, command=self.points_listbox.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self.points_listbox.config(yscrollcommand=scrollbar.set)
                
                # Display buy points
                self.points_listbox.insert(tk.END, "Buy Points:")
                for point in buy_points:
                    self.points_listbox.insert(tk.END, f"Date: {point[0]}, Price: {point[1]}")
                
                # Display sell points
                self.points_listbox.insert(tk.END, "")
                self.points_listbox.insert(tk.END, "Sell Points:")
                for point in sell_points:
                    self.points_listbox.insert(tk.END, f"Date: {point[0]}, Price: {point[1]}")

            except Exception as es:
                messagebox.showerror("Error",f"Due To :{str(es)}",parent=self.root)


    def news(self):
        if  self.var_Query.get()=="" or self.var_SortBy.get()=="" or self.var_No_Articles.get()=="" or self.var_API.get()=="":
            messagebox.showerror("Error","Not all field filled",parent=self.root)
        else:
            try:
                api_key = self.var_Api_key1.get()   #"use personal API key"
                main_url = "https://newsapi.org/v2/everything"
                params = {
                    "q": self.var_Query1.get(),
                    "language": "en",
                    "sortBy": self.var_SortBy1.get(),
                    "apiKey": api_key
                }

                no_of_articles = int(self.var_No_Articles1.get())

                news = requests.get(main_url, params=params).json()
                articles = news["articles"]

                # Creating a new window to displaying news
                news_window = tk.Toplevel(self.root)
                news_window.title("Stock News")
                news_text = tk.Text(news_window, wrap="word", width=80, height=30)
                news_text.pack(padx=10, pady=10, fill="both", expand=True)

                # Displaying the specified number of articles
                for index, article in enumerate(articles[:no_of_articles], start=1):
                    if article.get('title') and article.get('url'):
                        news_text.tag_configure(f"link_{index}", foreground="blue", underline=True)
                        news_text.insert(tk.END, f"{index}. {article['title']}\n", f"link_{index}")
                        news_text.tag_bind(f"link_{index}", "<Button-1>", lambda event, link=article['url']: webbrowser.open(link))
                        news_text.insert(tk.END, "\n")

                # Prevent editing
                news_text.config(state=tk.DISABLED)
            except Exception as es:
                messagebox.showerror("Error", f"Due To :{str(es)}", parent=self.root)



    def exit(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    obj = Stock(root)
    root.mainloop()
       


