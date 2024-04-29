import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of the page
st.title("LGBTQIA+ Representation in Television")
df_show_info = pd.read_csv("show info.csv")
st.dataframe(df_show_info)