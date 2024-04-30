import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from category_encoders import OrdinalEncoder
from sklearn.feature_selection import r_regression
st.set_option('deprecation.showPyplotGlobalUse', False)

df_cable = pd.read_excel("/mount/src/lgbtqia-rep/src/streamlit/data/cable.xlsx")
df_cable = df_cable.rename(columns={'Unnamed: 0': 'Characters'})
df_cable["Characters"] = df_cable["Characters"].str.replace(" Characters", "")

# GLAAD Analysis - Cable
st.header("LGBTQIA+ Representation in Cable TV :tv:")
st.dataframe(df_cable)

st.subheader("General Analysis")
# Plotting the data
cable = df_cable.T
new_header = cable.iloc[0] 
cable = cable[1:]
cable.columns = new_header 
sns.lineplot(cable).set(title='LGBTQ+ Characters on Cable TV')
plt.grid()
st.pyplot()

# Analysis
st.write("Cable TV has increased steadily their LGBTQ+ character representation since 2015, however, this has decreased starting from 2019 and onwards. The most represented group are gay men, followed by lesbians and bisexual characters. This gap has shortened throughout the years, nonetheless, it is worth noting that bisexual representation experienced a significant drop in 2018.") 
st.write("Regarding characters in the transgender spectrum, their numbers in cable TV have not surpassed 20 since GLAAD began reporting in 2010. There has also been a decrease starting from 2019.")
st.write("This setback has taken LGBTQ+ representation all the way back to 2016-2017 levels.")

st.subheader("Linear Regression")
cable = cable.reset_index()
cable = cable.rename(columns={'index':'Year'})
# Changing data types
cable['Year'] = cable['Year'].astype("object")
cable['Gay'] = cable['Gay'].astype("int64")
cable['Bisexual'] = cable['Bisexual'].astype("int64")
cable['Transgendered'] = cable['Transgendered'].astype("int64")
cable['Lesbian '] = cable['Lesbian '].astype("int64")
# Encoding the Year
encoder = OrdinalEncoder()
a = np.array(cable['Year']).reshape(-1,1)
cable['Year_encoded'] = encoder.fit_transform(a) 
cable
x = np.array(cable["Year_encoded"]).reshape(-1,1)
y_gay = np.array(cable["Gay"]).reshape(-1,1)
y_les = np.array(cable["Lesbian "]).reshape(-1,1)
y_bi = np.array(cable["Bisexual"]).reshape(-1,1)
y_trans = np.array(cable["Transgendered"]).reshape(-1,1)
st.write("Pearson Regression Coefficients for the gay, lesbian, bisexual and transgender characters in relation to the year:")
st.write(pd.DataFrame({
    'Characters': ['Gay', 'Lesbian', 'Bisexual', 'Transgendered'],
    'Regression Coefficient': [r_regression(x, y_gay.ravel()), r_regression(x, y_les.ravel()), r_regression(x, y_bi.ravel()), r_regression(x, y_trans.ravel())],
}))
st.write("The Pearson Coefficients indicate a high to very high linear correlation for all variables. Thus, we can proceed to create the models for each target variable:")

model_gay = LinearRegression()
model_gay.fit(x,y_gay)

model_les = LinearRegression()
model_les.fit(x,y_les)

model_bi = LinearRegression()
model_bi.fit(x,y_bi)

model_trans = LinearRegression()
model_trans.fit(x,y_trans)

gay_pred = model_gay.predict(x)
les_pred = model_les.predict(x)
bi_pred = model_bi.predict(x)
trans_pred = model_trans.predict(x)

st.write('Scores (how well the regression model fits the observed data) for the cable TV models:')
st.write(pd.DataFrame({
    'Characters': ['Gay', 'Lesbian', 'Bisexual', 'Transgendered'],
    'Scores': [model_gay.score(x,y_gay), model_les.score(x,y_les), model_bi.score(x,y_bi), model_trans.score(x,y_trans)],
}))
st.write('Based on these scores, we can expect the lesbian model to be the best performing one, followed by the bi, trans, and lastly gay models.')

st.subheader("Visualization")
sns.set_style("whitegrid")
fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(20, 4), sharey=True)
sns.regplot(x=x, y=y_gay, ax=axes[0]).set(title='Gay Characters on Cable TV', ylabel='Characters')
sns.regplot(x=x, y=y_les, ax=axes[1]).set(title='Lesbian Characters on Cable TV')
sns.regplot(x=x, y=y_bi, ax=axes[2]).set(title='Bisexual Characters on Cable TV')
sns.regplot(x=x, y=y_trans, ax=axes[3]).set(title='Transgendered Characters on Cable TV')
# Display the plot
st.pyplot(fig)

st.subheader("Prediction")
st.write('Predicting future years, where:')
st.write(pd.DataFrame({
    'Year': [2022, 2023, 2024, 2025, 2026],
    'Encoding': [13, 14,15,16,17],
}))
# Establishing the future years
x_future = [2022, 2023, 2024, 2025, 2026]
x_future = np.array(x_future).reshape(-1,1)
x_future = np.concatenate((np.array(cable['Year']).reshape(-1,1), x_future)).flatten()

# Encoding the future years
encoder = OrdinalEncoder()
x_future_encoded = np.array(x_future).reshape(-1,1).flatten()
x_future_encoded = encoder.fit_transform(x_future_encoded) 

# Predicting the future years
f_gay_pred = model_gay.predict(x_future_encoded).reshape(-1,1).flatten()
f_les_pred = model_les.predict(x_future_encoded).reshape(-1,1).flatten()
f_bi_pred = model_bi.predict(x_future_encoded).reshape(-1,1).flatten()
f_trans_pred = model_trans.predict(x_future_encoded).reshape(-1,1).flatten()

sns.set_style("whitegrid")
fig, axes = plt.subplots(ncols=4,nrows=1, figsize=(18,4), sharey=True)

sns.scatterplot(x=cable["Year"], y=cable["Gay"], ax=axes[0]).set(title='Gay Characters on Cable TV', ylabel='Characters')
sns.lineplot(x=x_future, y=f_gay_pred, ax=axes[0])

sns.lineplot(x=x_future, y=f_les_pred, ax=axes[1]).set(title='Lesbian Characters on Cable TV')
sns.scatterplot(x=cable["Year"], y=cable["Lesbian "], ax=axes[1])

sns.lineplot(x=x_future, y=f_bi_pred, ax=axes[2]).set(title='Bisexual Characters on Cable TV')
sns.scatterplot(x=cable["Year"], y=cable["Bisexual"], ax=axes[2])

sns.lineplot(x=x_future, y=f_trans_pred, ax=axes[3]).set(title='Transgendered Characters on Cable TV')
sns.scatterplot(x=cable["Year"], y=cable["Transgendered"], ax=axes[3])
st.pyplot(fig)

st.subheader("Comparison")
st.write('Predicted Characters for 2022:')
st.write(pd.DataFrame({
    'Characters': ['Gay', 'Lesbian', 'Bisexual', 'Transgendered'],
    'Predicted for 2022': [f_gay_pred[12].round(), f_les_pred[12].round(), f_bi_pred[12].round(), f_trans_pred[12].round()],
    'Actual data for 2022': [46,40,39,9],
}))

st.subheader("Conclusion")
st.write('The proposed linear regression returns much higher values than the actual reported ones in 2022.')
st.write('This is likely due to a bias in the model: although the representation of LGBTQ+ identities in cable TV was high in previous years, it has waned in recent years, which means it will take time for the representation to increase again.')
st.write('The linear regression model does not account for this setback, and expects a significant increase in the next years.')