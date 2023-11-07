import DataPreprocessing as DP

def MainStep():
    OrderDf, ProductDf = DP.PreProcessing()
    print(OrderDf.head(), ProductDf.head())

if __name__ == '__main__':
    MainStep()