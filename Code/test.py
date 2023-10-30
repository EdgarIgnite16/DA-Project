import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# Đọc file csv
elec_df = pd.read_csv("../Dataset/electronics.csv", encoding="ISO-8859-1")

# Xuất file csv
print(elec_df)

# Đếm số hàng có giá trị Null trong từng cột
print(elec_df.isnull().sum())
