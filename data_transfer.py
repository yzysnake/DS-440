import pandas as pd


class Category:
    def __init__(self, dataframe):
        # Initialize the class with the input dataframe
        self.dataframe = dataframe

    def createDataset(self):
        '''
        Main function to execute all other functions except SOD(),
        which add the technical indicators to the dataframe.
        '''
        self.RA_5()
        self.RA_10()
        self.MACD()
        self.CCI()
        self.ATR()
        self.BOLL()
        self.MA()
        self.MTM()
        self.ROC()
        self.WPR()

    def createDataset_SOD(self):
        '''
        Main function to execute all other functions,
        including the SOD function, which adds the technical
        indicators to the dataframe.
        '''
        self.RA_5()
        self.RA_10()
        self.MACD()
        self.CCI()
        self.ATR()
        self.BOLL()
        self.MA()
        self.MTM()
        self.ROC()
        self.WPR()
        self.SOD()

    def RA_5(self):
        '''
        Calculate the 5-day rolling average of standard deviation.
        Adds a new column 'RA_5' to the dataframe.
        '''
        self.dataframe['RA_5'] = self.dataframe['Adj Close'].rolling(window=5).std()

    def RA_10(self):
        '''
        Calculate the 10-day rolling average of standard deviation.
        Adds a new column 'RA_10' to the dataframe.
        '''
        self.dataframe['RA_10'] = self.dataframe['Adj Close'].rolling(window=10).std()

    def MACD(self):
        '''
        Calculate the Moving Average Convergence Divergence (MACD),
        which is the difference between the 12-day EMA and the 26-day EMA.
        Adds a new column 'MACD' to the dataframe.
        '''
        self.dataframe['MACD'] = pd.Series.ewm(self.dataframe['Adj Close'], span=12).mean() - \
                                 pd.Series.ewm(self.dataframe['Adj Close'], span=26).mean()

    def CCI(self):
        """
        Calculate the Commodity Channel Index (CCI) for the given data.
        Adds a new column 'CCI_n' to the dataframe, where n is the number of periods used in the calculation.
        """
        n = 20
        TP = (self.dataframe['High'] + self.dataframe['Low'] + self.dataframe['Close']) / 3
        CCI = pd.Series((TP - TP.rolling(n, min_periods=n).mean()) / (0.015 * TP.rolling(n, min_periods=n).std()),
                        name='CCI_' + str(n))
        self.dataframe = self.dataframe.join(CCI)

    def ATR(self):
        """
        Calculate the Average True Range (ATR) for the given data.
        Adds a new column 'ATR' to the dataframe.
        """
        n = 14
        i = 0
        TR_l = [0]
        while i < self.dataframe.index[-1]:
            TR = max(self.dataframe.loc[i + 1, 'High'], self.dataframe.loc[i, 'Close']) - \
                 min(self.dataframe.loc[i + 1, 'Low'], self.dataframe.loc[i, 'Close'])
            TR_l.append(TR)
            i = i + 1
        TR_s = pd.Series(TR_l)
        ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean(), name='ATR')
        self.dataframe = self.dataframe.join(ATR)

    def BOLL(self):
        """
        Calculate Bollinger Bands for the given data.
        Adds new columns 'BollingerUpper' and 'BollingerLower' to the dataframe.
        """
        MA = pd.Series(self.dataframe['Close'].rolling(window=20).mean())
        MSD = pd.Series(self.dataframe['Close'].rolling(window=20).std())
        b1 = MA + 2 * MSD
        B1 = pd.Series(b1, name='BollingerUpper')
        self.dataframe = self.dataframe.join(B1)

        b2 = MA - 2 * MSD
        B2 = pd.Series(b2, name='BollingerLower')
        self.dataframe = self.dataframe.join(B2)

    def MA(self):
        """
        Calculate the moving average for the given data.
        Adds new columns 'MA_n' to the dataframe, where n is the number of periods used in the calculation.
        """
        n = 5
        MA = pd.Series(self.dataframe['Adj Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
        self.dataframe = self.dataframe.join(MA)

        n = 10
        MA = pd.Series(self.dataframe['Adj Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
        self.dataframe = self.dataframe.join(MA)

    def MTM(self):
        """
        Calculate Momentum for the given data.
        Adds new columns 'Momentum_n' to the dataframe, where n is the number of periods used in the calculation.
        """
        n = 30
        M = pd.Series(self.dataframe['Adj Close'].diff(n), name='Momentum_' + str(n))
        self.dataframe = self.dataframe.join(M)

        n = 90
        M = pd.Series(self.dataframe['Adj Close'].diff(n), name='Momentum_' + str(n))
        self.dataframe = self.dataframe.join(M)

    def ROC(self):
        """
        Calculate the Rate of Change (ROC) for the given data.
        Adds a new column 'ROC' to the dataframe.
        """
        n = 30
        M = self.dataframe['Adj Close'].diff(n)
        N = self.dataframe['Adj Close'].shift(n)
        ROC = pd.Series(M / N, name='ROC')
        self.dataframe = self.dataframe.join(ROC)

    def WPR(self):
        n = 14
        WPR = pd.Series((self.dataframe['Close'] - self.dataframe['Low'].rolling(n).min()) /
                        (self.dataframe['High'].rolling(n).max() - self.dataframe['Low'].rolling(n).min()) * 100,
                        name="WPR_%s" % str(n))
        self.dataframe = self.dataframe.join(WPR)

    def SOD(self):
        """
        Calculate the Stochastic Oscillator D (SOD) for the given data.
        Adds a new column 'SOD_n' to the dataframe, where n is the number of periods used in the calculation.
        """
        n = 14
        SOk = pd.Series(
            (self.dataframe['Close'] - self.dataframe['Low']) / (self.dataframe['High'] - self.dataframe['Low']),
            name='SO%k')
        SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SOD_' + str(n))
        self.dataframe = self.dataframe.join(SOd)
