import pandas as pd
import matplotlib.pyplot as plt

def Preprocessing():
    # Đọc file csv
    maindf = pd.read_csv("../Dataset/electronics.csv", encoding="ISO-8859-1")
    
    # Tiền xử lý dữ liệu
    # Lọc: Loại bỏ các bộ dữ liệu bị thiếu trong cột brand
    maindf = maindf.dropna(subset=["brand"])
    maindf = maindf.fillna("None")
    maindf = maindf.reset_index(drop=True)

    return maindf


