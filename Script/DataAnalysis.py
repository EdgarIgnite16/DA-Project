import plotly.express as px

# hàm chính
def Analysis(dataframe):
    highestProfitMargins(dataframe)

# ==================================================== #
def highestProfitMargins(df):
    data = df
    fig = px.scatter(data, x="Wholesale Price", y="Retail Price", color="Product Category", hover_data=['Product Name'])
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