import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.feature_selection import r_regression
import plotly.graph_objects as go

show_info = pd.read_csv("data/show info.csv")
demographics = pd.read_csv("data/demographics.csv")

st.header("LGBTQIA+ Representation in Animated Shows :desktop_computer:")
st.subheader('Demographics')
st.dataframe(demographics)
st.write('As seen above, we have the data of LGBTQIA+ characters across a variety of animated shows. The data provided details their sexual orientation, gender, ethnic origin, role, type of representation (explicit or implicit) and show title. In total, there are 353 reported queer characters.')
st.write('The top ten shows with most queer characters are:')
show_rep = demographics.groupby(["show_title", "representation"]).size().unstack()
show_rep["total"] = show_rep["Explicit"].fillna(0) + show_rep["Implicit"].fillna(0)
top_ten = show_rep.nlargest(10, "total").head(10)
top_ten
st.write('Plotting the data shows that only a small portion of the reported LGBTQIA+ characters are actually explicit representations:')
top_ten.drop(["total"], axis=1).plot(kind="bar", cmap="Set2", xlabel="", ylabel="queer characters")
st.pyplot()

st.write('When grouping the shows by most explicit queer characters, we obtain the following rankings:')
show_rep.nlargest(10, "Explicit").drop(["total", "Implicit"], axis=1).plot(kind="bar", cmap="Pastel1", xlabel="", ylabel="queer characters", legend="")
st.pyplot()
st.write('While She-Ra and South Park remain in the first two positions, Steven Universe descends one place, BoJack Horseman and Big Mouth ascend a couple of places, and Summer Camp Island as well as The Loud House make their way into the graph.')
st.write(pd.DataFrame({
    'Type of Representation': ['Implicit', 'Explicit'],
    'Median of Overall Shows': [show_rep.median()[0], show_rep.median()[1]],
    'Median of Top 10 Shows': [top_ten.median()[0], top_ten.median()[1]],
}))
st.write('It is also worth noting that the top ten shows with explicit queer representations exceed the median of the total and explicit characters by 3 times, while it surpasses the implicit representation median by 6 times.')
st.write('(Due to the extreme differences in the data, the median is a more representative measure of central tendency).')
st.write(pd.DataFrame({
    'Type of Representation': ['Implicit', 'Explicit'],
    'Count': [demographics["representation"].value_counts()[0], demographics["representation"].value_counts()[1]],
}))
st.write('Amongst the 353 characters, 264 of them are explicit queer representations, while 89 of them are implicit.')

demographics['confirmation_date'] = pd.to_datetime(demographics['confirmation_date'])
demographics["year"] = demographics["confirmation_date"].dt.year
a = demographics.groupby(["year", "representation"]).size().unstack().rename(columns={'Explicit': 'Implicit', 'Implicit': 'Explicit'}) 
a.plot(kind="bar", title="Type of Representation by Year", ylabel="queer characters", cmap="seismic")
st.pyplot()

st.write('As the years go by, not only the LGBTQIA+ representation grows, but the representation also starts getting more explicit. While it started mainly with implicit queer characters, explicit representation begins growing steadily by the 2010s, with some rough patches.')
st.write('However, as the representation goes up, so do the implicit LGBTQIA+ characters. The only year with the shortest gap between explicit and implicit representation was 2020, which is not much considering that queer representation decreased all across the board during the pandemic.')
st.write('The increase in LGBTQIA+ representation at the cost of greater queer subtext is an undesirable outcome, considering it censors and obscures diversity. Progression should come not only in greater numbers, but also in greater visibility and clarity.')

demographics["orientation"].value_counts().sort_values(ascending=True).plot(kind="barh", title="Characters by Sexual/Romantic Orientation", ylabel="", xlabel="Characters", colormap="Pastel2")
st.pyplot()

st.write('Most of the characters represented in animated shows belong to the gay and lesbian groups. Following them is a sizable group of undetermined sexualities.')
st.write('The least represented groups in the LGBTQIA+ spectrum are the polyamorous, asexual and pansexual orientations. ')

demographics[demographics["orientation"].str.contains("Undetermined")].head(10)
st.write('Exploring the characters with undetermined sexualities allows us to shed more light on the categorization methods of the dataset. While Princess Bubblegum and Marceline fall into the sapphic spectrum (that is, women loving women), they are labeled as "Undetermined" because their sexuality may have not been confirmed textually in the show. ')
st.write('Thus, although at first hand the undetermined sexualities may seem like a censorship or invisibility issue, it’s rather a mattter of how they were labeled by the creators of the dataset. This is further supported by the fact that many of these characters fall into the explicit representation category. ')
st.write('Ultimately, a label is not required to communicate a queer story, and the type of representation is far more important. However, this shows an undersight in the categorization of the data, as a character’s sexuality can be described in other global labels.')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
fig.subplots_adjust(wspace=0.5)
demographics["gender"].value_counts().sort_values(ascending=True).plot(kind="barh", ax=axes[0], title="Gender of Queer Characters", ylabel="Gender")
demographics["gender"].value_counts().sort_values(ascending=True).drop(["Cis Man", "Cis Woman"]).plot(kind="barh", ax=axes[1], title="Gender Identities on the Trans Spectrum", ylabel="")
st.pyplot()

st.write('An analysis by gender reveals a substantial gap in the sexual and romantic orientations vs the gender diversity of animated characters. While lesbian and gay sexualities surpass 80 characters each, the top 2 trans identities barely have more than 15 characters. ')
st.write('Non-binary and trans women are the most represented groups within the trans spectrum. Meanwhile, intersex, undetermined and trans men are the least represented.')
st.write('It’s interesting to note that trans men, an identity aligned with the gender binary, is less represented than genderfluid, agender and non-binary folks; who defy the established gender structure.')

demographics["race"].value_counts().sort_values(ascending=True).plot(kind="bar", title="Ethnicity of Queer Characters", xlabel="", ylabel="", colormap="viridis")
st.pyplot()
st.write('A sustantial amount of queer characters are white (153) while people of colour (POC) amount to 84 characters.')

demographics["role"].value_counts().sort_values(ascending=False).to_frame()
st.write(pd.DataFrame({
    'Character Role': ['Recurring Character', 'Main Character', 'Supporting Character', 'Guest Character'],
    'Quantity': [demographics["role"].value_counts().sort_values(ascending=False).iloc[0], demographics["role"].value_counts().sort_values(ascending=False).iloc[1], demographics["role"].value_counts().sort_values(ascending=False).iloc[2], demographics["role"].value_counts().sort_values(ascending=False).iloc[3]],
}))
st.write('Mostly of the LGBTQIA+ characters reported in this dataset are recurring characters (175, almost half of them) while only 95 characters are protagonists. ')


st.subheader('General')
st.dataframe(show_info)
st.write('While the demographic dataset contains 353 entries (that is, 353 characters), the show information contains only 118. This means that there’s 118 unique shows in which the recorded LGBTQIA+ characters appear in.')

show_info.describe()
st.write('The above data shows the votes, rating, seasons, episodes and airing duration of the shows. ')
st.write('Once again, taking the median (Q2) as the reference due to the high variations in data, the median show lasted 3 seasons, had 33 episodes, had an airing duration of 897 hours, and a rating of 7.4. ')

rating = show_info["TV_rating"].value_counts().sort_values(ascending=False).to_frame()
trating = rating.transpose()
rating.plot(kind="bar", title="TV Rating of Queer Shows", xlabel="", colormap="plasma", legend="")
st.pyplot()
under_18 = trating["TV-Y"] + trating["TV-Y7"] + trating["TV-G"] + trating["TV-PG"] + trating["TV-14"]
over_18 = trating["TV-MA"]
st.write(pd.DataFrame({
    'Ratings': ['Under 18', 'Over 18'],
    'Quantity': [int(under_18.iloc[0]), int(over_18.iloc[0])],
}))

st.write('The majority of recorded TV shows belong to the under 18 years-old TV ratings (77), while only 35 of them are for mature audiences. This is to be expected since the dataset focuses on animated shows, which tend to be produced for younder audiences. ')
st.write('This is most likely to have an impact in the implicit and explicit LGBTQIA+ representations, since the topic is heavily guarded upon when it comes to children’s media. However, this is also encouraging. Since younger audiences are coming into contact with queer identities, they are more likely to have greater acceptance and hopefully discover their own sexuality and gender identity through the characters represented.')

a = pd.merge(demographics, show_info.drop('show_title', axis=1), on='ID').drop('ID', axis=1)
a['confirmation_date'] = pd.to_datetime(a['confirmation_date'])
a["year"] = a["confirmation_date"].dt.year
rating_order = {'TV-Y':'Under 18', 'TV-Y7':'Under 18', 'TV-G':'Under 18', 'TV-PG':'Under 18', 'TV-14':'Under 18', 'TV-MA':'Over 18'}
a['18_rating'] = a['TV_rating']
a["18_rating"] = a['18_rating'].map(rating_order)
a = a.groupby(['18_rating', 'representation']).size().unstack().rename(columns={'Explicit': 'Implicit', 'Implicit': 'Explicit'}) 
a.plot(kind="bar", cmap="Set2", xlabel="", ylabel="queer characters")
st.pyplot()

st.write(pd.DataFrame({
    'TV Ratings': ['Under 18', 'Over 18'],
    'Explicit/Implicit Ratio': [a.iloc[1,1]/a.iloc[1,0], a.iloc[0,1]/a.iloc[0,0]],
}))
st.write('Surprisingly, the ratio of implicit-explicit representation is greater in under 18 rated shows (0.40) than in over 18 rated shows (0.21).')
st.write('This means that for every 10 implicit representations in children and adolescent shows, there are 4 explicit characters. Meanwhile, in TV shows rated for adults, for every 10 implicit representations, there are 2 explicit LGBTQIA+ characters.')

show_info["IMDB_rating"].plot(kind="hist", title="IMDB Ratings of Queer Shows", xlabel="Rating", colormap="Set2")
st.pyplot()
st.write('The IMDB ratings of queer shows are extremely positive, most of them located above the 6 rating. This indicates a great reception of the media, whcih favours the messages and characters reproduced in the show.')

show_info["network"].value_counts().head(10).plot(kind="bar", title="Top 10 Networks that Produce Queer Shows", xlabel="")
st.pyplot()
st.write('Netflix, Cartoon Network and Fox are the top 3 networks that produce queer shows. These three have produced over 10 different queer shows, in comparison to the rest who have barely made it past 5 shows.')



st.subheader('Encoding the Year Variables')
demographics = demographics.drop(columns='year')
df = pd.merge(demographics, show_info.drop('show_title', axis=1), on='ID').drop('ID', axis=1)
# Creating a year column
df["year"] = df["confirmation_date"].dt.year
# Dropping the confirmation, start and end dates
df = df.drop(columns=['confirmation_date', 'start_date', 'end_date'])
st.dataframe(df)

st.write('Encoding the year:')
queer_characters = df.groupby(['year', 'orientation']).size().unstack().drop('Straight', axis=1).fillna(0).sum(axis=1)
queer_characters = pd.DataFrame(queer_characters, columns=['Queer Characters'])
queer_characters = queer_characters.reset_index()

queer_characters['year'] = queer_characters['year'].astype("object")
encoder = OrdinalEncoder()
a = np.array(queer_characters['year']).reshape(-1,1)
queer_characters['year_encoded'] = encoder.fit_transform(a) 
st.dataframe(queer_characters)

trans_characters = df.groupby(['year', 'gender']).size().unstack().drop(['Cis Man', 'Cis Woman'], axis=1).fillna(0).sum(axis=1)
trans_characters = pd.DataFrame(trans_characters, columns=['Trans Characters'])
trans_characters = trans_characters.reset_index()

trans_characters['year'] = trans_characters['year'].astype("object")
a = np.array(trans_characters['year']).reshape(-1,1)
trans_characters['year_encoded'] = encoder.fit_transform(a) 

st.subheader('Linear Regressions')
x_queer = np.array(queer_characters['year_encoded']).reshape(-1,1)
y_queer = np.array(queer_characters['Queer Characters']).reshape(-1,1)

x_trans = np.array(trans_characters['year_encoded']).reshape(-1,1)
y_trans = np.array(trans_characters['Trans Characters']).reshape(-1,1)

st.write('The Pearson Regression Coefficients for the queer and trans variables are:')
st.write(pd.DataFrame({
    'Characters': ['Queer', 'Transgender'],
    'Regression Coefficient': [r_regression(x_queer, y_queer.ravel()), r_regression(x_trans, y_trans.ravel())]
}))

model_queer = LinearRegression()
model_queer.fit(x_queer,y_queer)

model_trans = LinearRegression()
model_trans.fit(x_trans,y_trans)

queer_pred = model_queer.predict(x_queer)
trans_pred = model_trans.predict(x_trans)

st.write('Scores for the models:')
st.write(pd.DataFrame({
    'Characters': ['Queer', 'Transgender'],
    'Model Scores': [model_queer.score(x_queer,y_queer), model_trans.score(x_trans,y_trans)]
}))
st.write('The model scores are not very high.')

sns.set_style("whitegrid")
fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(18,4), sharey=True)
sns.regplot(x=x_queer, y=y_queer, ax=axes[0]).set(title='Queer Characters on Animated Shows', ylabel='Characters')
sns.regplot(x=x_trans, y=y_trans, ax=axes[1]).set(title='Trans Characters on Animated Shows')
st.pyplot()



st.subheader('Clustering Model')
st.write('To do a clustering model of the data, first we have to encode the categorical values. We can either do this by ordinal encoding or one-hot encoding. Because some features have multiple categorical values attached to them, such as network and genre, I’m gonna apply a one-hot encoding, while the rest of the categorical values will have a ordinal encoding.')
st.write('Although the one-hot encodings will cause an increase in dimensionality, the multiple values make it hard to ordinal encode the data. However, for the data that follows a clear hierarchy, I will use a ordinal encode.')

df_encoded = df.copy()
df_encoded = df_encoded.drop(columns=['show_title', 'character_name'])
st.dataframe(df_encoded)

# Encoding networks
network = df_encoded["network"].str.split(", ", expand=True)
encoder = OneHotEncoder(return_df=True,use_cat_names=True)
network_encoded = encoder.fit_transform(network)
network_encoded.columns = [col.split("_")[-1] for col in network_encoded.columns]
network_encoded = network_encoded.drop(columns=["nan"])
network_encoded = network_encoded.T.groupby(network_encoded.columns).sum().T
df_encoded = pd.concat([df_encoded, network_encoded], axis=1)
df_encoded = df_encoded.drop(columns=['network'])

# Encoding genre
genre = df_encoded["genre"].str.split(", ", expand=True)
genre_encoded = encoder.fit_transform(genre)
genre_encoded.columns = [col.split("_")[-1] for col in genre_encoded.columns]
genre_encoded = genre_encoded.drop(columns=["nan"])
genre_encoded = genre_encoded.T.groupby(genre_encoded.columns).sum().T
df_encoded = pd.concat([df_encoded, genre_encoded], axis=1)
df_encoded = df_encoded.drop(columns=['genre'])

# Encoding race, gender and orientation
df_encoded['race'] = df_encoded['race'].fillna('Undetermined_race')
df_encoded['gender'] = df_encoded['gender'].str.replace('Undetermined', 'Undetermined_gender')
df_encoded['orientation'] = df_encoded['orientation'].str.replace('Undetermined', 'Undetermined_orientation')
encoder = OneHotEncoder(cols=['race', 'gender', 'orientation'],handle_unknown='return_nan',return_df=True,use_cat_names=True)
df_encoded = encoder.fit_transform(df_encoded)
df_encoded.columns = [col.split("race_")[-1] for col in df_encoded.columns]
df_encoded.columns = [col.split("gender_")[-1] for col in df_encoded.columns]
df_encoded.columns = [col.split("orientation_")[-1] for col in df_encoded.columns]

# Ordinal Encodings
role_order = {'Guest Character':0, 'Supporting Character':1, 'Recurring Character':2, 'Main Character':3}
df_encoded['role'] = df_encoded['role'].map(role_order)

rep_order = {'Implicit':0, 'Explicit':1}
df_encoded['representation'] = df_encoded['representation'].map(rep_order)

rating_order = {'none listed':0, 'TV-Y':1, 'TV-Y7':2, 'TV-G':3, 'TV-PG':4, 'TV-14':5, 'TV-MA':6}
df_encoded['TV_rating'] = df_encoded['TV_rating'].map(rating_order)

st.dataframe(df_encoded)
st.write('All values in the dataframe are numerical, thus we can proceed to normalize the data.')

st.subheader('Normalization')
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_encoded)
df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns)
st.dataframe(df_scaled)


st.subheader('KMeans++ Cluster Algorithm')
kmeans = KMeans(init='k-means++', random_state=23)
kmeans.fit(df_scaled)
clusters = kmeans.labels_
x_embedded_2d = TSNE(n_components=2, learning_rate='auto', random_state=74).fit_transform(df_scaled)
x_embedded_3d = TSNE(n_components=3, learning_rate='auto', random_state=74).fit_transform(df_scaled)
st.write(pd.DataFrame({
    'Scores': ['Silhouette', 'Davies Bouldin'],
    '': [silhouette_score(x_embedded_2d, kmeans.fit_predict(x_embedded_2d)),  davies_bouldin_score(x_embedded_2d, clusters)]
}))
st.write('The silhouette score for the clustering model is 0.3602, values closest to 1 are preferrable. This is not a bad score.')
st.write('While in the davies boulding score, values closest to 0 are preferrable. A 1.28 is an acceptable score.')

plt.figure(figsize=(6,6))
sns.scatterplot(
    x= x_embedded_2d[:, 0], y=x_embedded_2d[:, 1],
    hue=clusters,
    palette=sns.color_palette("twilight",8),
    data=df_scaled,
    alpha=0.7
    )
plt.title('TSNE of KMeans++ Clustering 2D')
plt.legend(title='Clusters') 
st.pyplot()

ax = plt.figure(figsize=(6,6)).add_subplot(projection = '3d')
ax.scatter(
    xs= x_embedded_3d[:, 0], 
    ys= x_embedded_3d[:, 1],
    zs= x_embedded_3d[:, 2],
    c=clusters,
    cmap='twilight',
    alpha=0.8,
)
plt.title('TSNE of KMeans++ Clustering 3D')
st.pyplot()


# Zip feature names with t-SNE data
custom_data = list(zip(df_scaled.columns, x_embedded_2d))

# Define hover template
hover_template = '<b>Feature:</b> %{customdata[0]}'

# Plot t-SNE visualization with hover labels
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_embedded_2d[:, 0], 
    y=x_embedded_2d[:, 1], 
    mode='markers',
    marker=dict(color=clusters),
    customdata=custom_data,
    hovertemplate=hover_template,
))
fig.update_layout(width=600, height=600)
st.plotly_chart(fig, theme=None)