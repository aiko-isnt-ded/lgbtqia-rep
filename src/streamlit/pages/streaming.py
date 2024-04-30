import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from category_encoders import OrdinalEncoder
from sklearn.feature_selection import r_regression
st.set_option('deprecation.showPyplotGlobalUse', False)

df_streaming = pd.read_excel("data/streaming.xlsx")
df_streaming = df_streaming.rename(columns={'Unnamed: 0': 'Characters'})
df_streaming["Characters"] = df_streaming["Characters"].str.replace(" Characters", "")

# GLAAD Analysis - Cable
st.header("LGBTQIA+ Representation in Streaming TV Services :movie_camera:")
st.dataframe(df_streaming)

st.subheader("General Analysis")
# Plotting the data
streaming = df_streaming.T
new_header = streaming.iloc[0] 
streaming = streaming[1:]
streaming.columns = new_header 
sns.lineplot(streaming).set(title='LGBTQ+ Characters on Streaming TV')
plt.grid()
st.pyplot()

# Analysis
st.write("While streaming services data started being quantified in 2015, 5 years later than that of cable TV, it’s numbers are already bigger than that of of cable in the aforementioned years. ") 
st.write("The gap between represented identities in streaming TV is historically shorter than the gap in cable TV, without dramatical drops and long recovery times. Although all identities experienced a reduced representation in 2020, this can likely be attributed to the COVID-19 pandemic, and the numbers quickly picked up again in the next year in an all-time high.")
st.write("Even if the lesbian, gay and bisexual representation numbers are commendable, surpassing 100 unlike cable TV, characters in the transgender spectrum are still left behind: it wasn’t until 2021 that more than 20 transgendered characters were shown in streaming services.")
st.write("The advantages streaming services have over cable TV in regards to LGBTQ+ representation is likely due to their private sector investment and target market. Since this gives them more room to work around censorship, the themes and characters they explore get to be more varied.")

st.subheader("Linear Regression")
streaming = streaming.reset_index()
streaming = streaming.rename(columns={'index':'Year'})
# Changing data types
streaming['Year'] = streaming['Year'].astype("object")
streaming['Gay'] = streaming['Gay'].astype("int64")
streaming['Bisexual'] = streaming['Bisexual'].astype("int64")
streaming['Transgendered'] = streaming['Transgendered'].astype("int64")
streaming['Lesbian '] = streaming['Lesbian '].astype("int64")
# Encoding the Year
encoder = OrdinalEncoder()
a = np.array(streaming['Year']).reshape(-1,1)
streaming['Year_encoded'] = encoder.fit_transform(a) 
streaming
x = np.array(streaming["Year_encoded"]).reshape(-1,1)
y_gay = np.array(streaming["Gay"]).reshape(-1,1)
y_les = np.array(streaming["Lesbian "]).reshape(-1,1)
y_bi = np.array(streaming["Bisexual"]).reshape(-1,1)
y_trans = np.array(streaming["Transgendered"]).reshape(-1,1)
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

st.write('Scores (how well the regression model fits the observed data) for the streaming TV models:')
st.write(pd.DataFrame({
    'Characters': ['Gay', 'Lesbian', 'Bisexual', 'Transgendered'],
    'Scores': [model_gay.score(x,y_gay), model_les.score(x,y_les), model_bi.score(x,y_bi), model_trans.score(x,y_trans)],
}))
st.write('The scores of these models are higher than the cable ones. Based on the scores, we can expect the gay model to be the best performing one.')

st.subheader("Visualization")
sns.set_style("whitegrid")
fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(20, 4), sharey=True)
sns.regplot(x=x, y=y_gay, ax=axes[0]).set(title='Gay Characters on Streaming TV', ylabel='Characters')
sns.regplot(x=x, y=y_les, ax=axes[1]).set(title='Lesbian Characters on Streaming TV')
sns.regplot(x=x, y=y_bi, ax=axes[2]).set(title='Bisexual Characters on Streaming TV')
sns.regplot(x=x, y=y_trans, ax=axes[3]).set(title='Transgendered Characters on Streaming TV')
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
x_future = np.concatenate((np.array(streaming['Year']).reshape(-1,1), x_future)).flatten()

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

sns.scatterplot(x=streaming["Year"], y=streaming["Gay"], ax=axes[0]).set(title='Gay Characters on Streaming TV', ylabel='Characters')
sns.lineplot(x=x_future, y=f_gay_pred, ax=axes[0])

sns.lineplot(x=x_future, y=f_les_pred, ax=axes[1]).set(title='Lesbian Characters on Streaming TV')
sns.scatterplot(x=streaming["Year"], y=streaming["Lesbian "], ax=axes[1])

sns.lineplot(x=x_future, y=f_bi_pred, ax=axes[2]).set(title='Bisexual Characters on Streaming TV')
sns.scatterplot(x=streaming["Year"], y=streaming["Bisexual"], ax=axes[2])

sns.lineplot(x=x_future, y=f_trans_pred, ax=axes[3]).set(title='Transgendered Characters on Streaming TV')
sns.scatterplot(x=streaming["Year"], y=streaming["Transgendered"], ax=axes[3])
st.pyplot(fig)

st.subheader("Comparison")
st.write('Predicted Characters for 2022:')
st.write(pd.DataFrame({
    'Characters': ['Gay', 'Lesbian', 'Bisexual', 'Transgendered'],
    'Predicted for 2022': [f_gay_pred[7].round(), f_les_pred[7].round(), f_bi_pred[7].round(), f_trans_pred[7].round()],
    'Actual data for 2022': [130,107,84,16],
}))

st.subheader("Conclusion")
st.write('This time, the model underpredicted the LGBTQ+ characters, likely because in the last year (2022) sexual diversity on streaming services skyrocketed, and the model had little warning to estimate the representation growth.')
st.write('Bisexual characters were the best predicted this time, which is surprising since the increase they experienced is by far the most dramatic one.')
st.write('Trans characters predictions come close to the real numbers in both models, likely because the growth for this identity is not as steep as with the other groups: instead, it’s more gradual, allowing the model to predict accurately the behaviour of the representation.')
st.write('Ultimately, both the cable and streaming TV models could be improved, specially the cable one. These models show how volatile LGBTQ+ representation is on TV, and hints that there may be more factors at play when it comes to it.')