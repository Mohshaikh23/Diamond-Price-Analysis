import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv("diamonds.csv")
data = data.drop("Unnamed: 0", axis=1)
data["size"] = data["x"] * data["y"] * data["z"]
data["cut"] = data["cut"].map({"Ideal": 1, "Premium": 2, "Good": 3, "Very Good": 4, "Fair": 5})

# Split the data
x = np.array(data[["carat", "cut", "size"]])
y = np.array(data[["price"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor()
model.fit(xtrain, ytrain)

# Streamlit app
st.title("Diamond Price Prediction App")
st.sidebar.title("Input Features")

carat = st.sidebar.number_input("Carat Size", min_value=0.2, max_value=5.01, step=0.01, value=1.0)
cut = st.sidebar.selectbox("Cut Type", ["Ideal", "Premium", "Good", "Very Good", "Fair"])
size = st.sidebar.number_input("Size", min_value=0.1, max_value=1000.0, step=0.1, value=10.0)

cut_mapping = {"Ideal": 1, "Premium": 2, "Good": 3, "Very Good": 4, "Fair": 5}
cut_encoded = cut_mapping[cut]

features = np.array([[carat, cut_encoded, size]])
predicted_price = model.predict(features)

st.sidebar.markdown("Predicted Diamond's Price:")
st.sidebar.write("$", predicted_price[0])

# Data Visualization
st.header("Data Visualization")

# Scatter plot with size vs price
scatter_fig = px.scatter(data_frame=data, x="size", y="price", size="size", color="cut", trendline="ols")
st.plotly_chart(scatter_fig)

# Box plots
box_fig1 = px.box(data, x="cut", y="price", color="color")
box_fig2 = px.box(data, x="cut", y="price", color="clarity")

st.plotly_chart(box_fig1)
st.plotly_chart(box_fig2)

# Exclude non-numeric columns from correlation calculation
numeric_columns = ["price", "carat", "size"]
numeric_data = data[numeric_columns]

# Calculate correlation matrix
correlation = numeric_data.corr()
st.write("Correlation with Price:")
st.write(correlation["price"].sort_values(ascending=False))
