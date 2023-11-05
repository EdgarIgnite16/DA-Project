import pandas as pd
import matplotlib.pyplot as plt

def PreProcessing():
    # Đọc file csv
    OrderDf = pd.read_csv("../Dataset/orders.csv")
    
    # ========================================================= #
    # Tiền xử lý dữ liệu trên file Order.csv
    # 1. Kiểm tra tập dữ liệu
    # OrderDf.info() # Kiểm tra thông tin tập dữ liệu
    # a = OrderDf.head() # Kiểm tra sơ bộ tập dữ liệu
    # b = OrderDf.isna().sum() # Kiểm tra xem data có trống hay không
    # c = OrderDf.isnull().sum() # Kiểm tra xem data có NULL hay không
    # print(a, b, c) # In kết quả kiểm tra ra màn hình 

    # 2. Tiền xử lý dữ liệu
    # Biến đổi dữ liệu: Chuyển đổi dạng dữ liệu ngày thành ngày và giờ trên cột: 'Date Order was placed' và 'Delivery Date'
    OrderDf['Date Order was placed'] = pd.to_datetime(OrderDf['Date Order was placed'], format='%d-%b-%y')
    OrderDf['Delivery Date'] = pd.to_datetime(OrderDf['Delivery Date'], format='%d-%b-%y')
    # print(df.head()) # Kiểm tra lại tập dữ liệu sau khi đã xử lý xong

    # Biến đổi dữ liệu: Tiễn hành tiền xử lý các lỗi viết in hoa trong cột "Customer Status" và ép kiểu dữ liệu về dạng String
    OrderDf['Customer Status'] = OrderDf['Customer Status'].str.lower()
    OrderDf['Customer Status'] = OrderDf['Customer Status'].str.capitalize()
    OrderDf['Customer Status'] = OrderDf['Customer Status'].astype('string')
    # print(df['Customer Status'].unique()) # Kiểm tra lại 1 lần nữa

    # Tạo dữ liệu mới: Tạo một cột mới tên là Item Retail Value để tính toán lại giá bán lẻ trên từng mặt hàng
    OrderDf['Item Retail Value'] = OrderDf['Total Retail Price for This Order']/OrderDf['Quantity Ordered']
    # print(df.head()) # Kiểm tra lại bộ dữ liệu

    # Tạo 1 dataframe mới: Phân nhóm giá trị trung bình theo Sản phẩm - Giá vốn trên mỗi đơn vị và Giá trị bán lẻ mặt hàng và đếm cả số lượng sản phẩm đó
    counts = OrderDf.groupby('Product ID')
    dfAvg = counts.size().to_frame(name='N Rows')
    dfAvg = dfAvg.join(counts.agg({'Cost Price Per Unit': 'mean'}))
    dfAvg = dfAvg.join(counts.agg({'Item Retail Value': 'mean'}))
    dfAvg = dfAvg.reset_index()
    # print(dfAvg.head()) # Kiếm tra

    # ========================================================= #
    # Nối các bảng dữ liệu từ 2 dataframe lại với nhau: 1 dataframe tính trung bình dựa trên sản phẩm và 1 dataframe về thông tin của sản phẩm
    # => Tạo thành 1 dataframe tổng thể về mặt nội dung cần phân tích
    dfProd = pd.read_csv('../Dataset/product-supplier.csv') # Đọc dữ liệu từ file product
    ProductDf = pd.merge(dfAvg, dfProd, on='Product ID', how='inner') # Nối dữ liệu 

    # Lọc dữ liệu: Tiến hành loại bỏ các cột không cần thiết trong việc phân tích dữ liệu sản phẩm
    ProductDf = ProductDf.drop(columns=['Product Line', 'Product Group', 'Supplier Country', 'Supplier Name', 'Supplier ID'])
    # Tiến hành đổi lại tên các cột sao cho phù hợp với bối cảnh phân tích bài toán
    ProductDf = ProductDf.rename(columns={'N Rows': 'Total Sold', 'Cost Price Per Unit': 'Wholesale Price', 'Item Retail Value': 'Retail Price'})
    # print(ProductDf.head()) # Kiểm tra

    return OrderDf, ProductDf

# running
PreProcessing()