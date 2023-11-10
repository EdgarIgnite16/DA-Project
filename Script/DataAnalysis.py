import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import metrics

# hàm chính
def Analysis(dataframe):
    # Hiển thị thông tin dataframe thông qua biểu đồ
    highestProfitMargins(dataframe)
    productLagestAndSmallest(dataframe)
    pricesVaryWithinCategories_WS(dataframe)
    pricesVaryWithinCategories_Re(dataframe)
    describeDF(dataframe) # Mô tả thông số thống kê

    # Tiến hành phân tích tập dữ liệu:
    # showGraph(dataframe)
    # LNR1(dataframe)
    # LNR2(dataframe)


# ==================================================== #
# Hiển thị biểu đồ tỷ suất lợi nhuận
def highestProfitMargins(df):
    fig = px.scatter(df, x="Wholesale Price", y="Retail Price", color="Product Category", hover_data=['Product Name'])
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey')
    fig.update_yaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey')

    fig.show()

# Hiển thị số lượng loại sản phẩm và tổng tiền bán được từ bé đến lớn
def productLagestAndSmallest(df):
    fig = px.histogram(df, x="Product Category", y='Total Sold', color="Product Category")
    fig.update_layout(xaxis={'categoryorder':'total descending'}, plot_bgcolor='white')
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.show()

# Hiển thị số lượng sản phẩm bán được dao động trong các khoảng mức giá khác nhau
# Wholesale - Bán sỉ
def pricesVaryWithinCategories_WS(df): 
    fig = px.histogram(df, x="Wholesale Price", y="Total Sold", marginal="box", hover_data=df.columns)
    fig.update_traces(marker_line_width=0.1,marker_line_color="white")
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.show()

# Retail - Bán lẻ
def pricesVaryWithinCategories_Re(df): 
    fig = px.histogram(df, x="Retail Price", y="Total Sold", marginal="box", hover_data=df.columns)
    fig.update_traces(marker_line_width=0.1,marker_line_color="white")
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.show()

# Mô tả dữ liệu thống kê
def describeDF(df):
    print(df['Wholesale Price'].describe())
    print(df['Retail Price'].describe())

# ==================================================== #
# Phân tích dữ liệu
def showGraph(df): 
    sns.lmplot(x="Wholesale Price", y="Retail Price", data=df, line_kws={'color': 'red'}) # Hiển thị biểu đồ dạng
    sns.pairplot(df) # Hiển thị các biểu đồ của từng cột
    plt.show() # Hiển thị

def LNR1(df):
    # Kiểm thử và huấn luyện mô hình
    # Huấn luyện trên 2 biến: Wholesale Price và Total Sold
    y = df["Retail Price"] # Gọi y là biến cần tìm 
    X = df[['Wholesale Price', 'Total Sold']] # Gọi x là mối tương quan cần tìm kiếm
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Bắt đầu huấn luyện mô hình
    lm2 = LinearRegression()
    lm2.fit(X_train,y_train)
    print(lm2.coef_) # in kết quả huấn luyện

    # Dự đoán dữ liệu thử nghiệm
    predictions = lm2.predict(X_test)
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Y Test (True Values)")
    plt.ylabel("Predicted Values")

    # Đánh giá mô hình
    print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
    print("MSE: ", metrics.mean_squared_error(y_test, predictions))
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    r2_lin2v = metrics.explained_variance_score(y_test, predictions) # hàm tính điểm số hồi quy (1.0 là cao nhất)
    print("Điểm số hồi quy: ", r2_lin2v) 
    # => Vậy mô hình rất tốt và phù hợp
    
    # Dư lượng
    sns.displot((y_test - predictions), kde=True, bins=50)
    plt.show() # Hiển thị

    # Hệ số
    cdf = pd.DataFrame(lm2.coef_,X.columns, columns=["Coefficient"])
    print(cdf) 
    # => Vậy không có mối tương quan với cột Total Sold, loại bỏ nó ra khỏi dữ liệu phân tích
    # => Lặp lại điều tương tự mà không có hệ số đó

def LNR2(df):
    # Kiểm thử và huấn luyện mô hình
    # Huấn luyện trên 1 biến: Wholesale Price
    y = df["Retail Price"] # Gọi y là biến cần tìm 
    X = df[['Wholesale Price']] # Gọi x là mối tương quan cần tìm kiếm
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    # Bắt đầu huấn luyện mô hình
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    print(lm.coef_)

    # Dự đoán dữ liệu thử nghiệm
    predictions1 = lm.predict(X_test)
    sns.scatterplot(x=y_test, y=predictions1)
    plt.xlabel("Y Test (True Values)")
    plt.ylabel("Predicted Values")

    # Đánh giá mô hình
    print("MAE: ", metrics.mean_absolute_error(y_test, predictions1))
    print("MSE: ", metrics.mean_squared_error(y_test, predictions1))
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions1)))

    r2_lin1v = metrics.explained_variance_score(y_test, predictions1) # hàm tính điểm số hồi quy (1.0 là cao nhất)
    print("Điểm số hồi quy: ", r2_lin1v) 

    # Dư lượng
    sns.displot((y_test - predictions1), kde=True, bins=50)
    plt.show() # Hiển thị

    # Hệ số
    cdf = pd.DataFrame(lm.coef_,X.columns, columns=["Coefficient"])
    print(cdf) 
