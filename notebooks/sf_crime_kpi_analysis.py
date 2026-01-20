#!/usr/bin/env python
# coding: utf-8

# ## KPI-driven analysis of urban crime to support resource allocation (San Francisco, 2016)

# ## 1. Objectives: 
# Investigating crime patterns in San Francisco in 2016 to identify the potential high-risk areas, time-ranges and crime types, with the aim of supporting the police resource allocation decisions.   

# In[ ]:


# Import Python libraries:
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium  # for creating interactive Maps
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Dataset and Method: 
# San Francisco Police Department Incidents for the year 2016 - [Police Department Incidents](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry) from San Francisco public data portal. Address and location has been anonymized by moving to mid-block or to an intersection. 
# You can download it here: 
# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Police_Department_Incidents_-_Previous_Year__2016_.csv  (dataset included in the IBM Data Science Professional Certificate)

# In[ ]:


# read  into a pandas dataframe:
df_incidents = pd.read_csv("Police_Department_Incidents_2016.csv")
display(df_incidents.head())


# In[ ]:


# check missing values
print(df_incidents.isnull().sum()) 


# In[ ]:


print(df_incidents.info())


# In[ ]:


df_incidents.shape


# Extracting a random sample of 10000 records is useful to reduce computational load and improve performance during exploratory analysis and visualization.

# In[ ]:


# extract a simple random sample without replacement for exploratory purpose
sample_sf=df_incidents.sample(n=10000,replace=False,random_state=1) # n=number of elements to extract 
# replace= False, the elements are not repeated; random_state= ensures we get the same sample if we run it multiple times. 

# save file to excel
sample_sf.to_excel("incidents_SF.xlsx", index=False)
# read the new Dataset:
df_sample=pd.read_excel("incidents_SF.xlsx")
display(df_sample.head())

df_sample.shape


# Each row consists of 13 features:
# > 1. **IncidntNum**: Incident Number
# > 2. **Category**: Category of crime or incident
# > 3. **Descript**: Description of the crime or incident
# > 4. **DayOfWeek**: The day of week on which the incident occurred
# > 5. **Date**: The Date on which the incident occurred
# > 6. **Time**: The time of day on which the incident occurred
# > 7. **PdDistrict**: The police department district
# > 8. **Resolution**: The resolution of the crime in terms whether the perpetrator was arrested or not
# > 9. **Address**: The closest address to where the incident took place
# > 10. **X**: The longitude value of the crime location 
# > 11. **Y**: The latitude value of the crime location
# > 12. **Location**: A tuple of the latitude and the longitude values
# > 13. **PdId**: The police department ID

# ## 3. KPI Analysis
# ### KPI 1: Crime Incidents Volume
# It measures the number of crime incidents reported in San Francisco in 2016 

# In[ ]:


total_crimes=len(df_sample)
print("The number of total crimes is:", total_crimes)


# This is the starting point for all the further calculations and comparisons. It can be used to detect evident increases across specific Districts or time periods and to support resource planning

# ### KPI 2: Top Crime Category Share

# In[ ]:


top_category=df_sample['Category'].value_counts().idxmax()
top_category_count=df_sample['Category'].value_counts().max()
top_category_share=(top_category_count/total_crimes)*100
print(f"Top category share is:{top_category_share:.2f}")


# Almost 27% of total crimes belong to Larceny/Theft, making it the dominant crime type.
# This suggests a prevalence of opportunistic crimes related to commercial areas, peak activity hours and social gatherings. 
# * Increase visible patrols in commercial and high-traffic areas, especially during peak hours

# ### KPI 3: Peak Crime Day

# In[ ]:


# Separate weekday from weekend 
weekday_map = {
    'Monday': 'Weekday',
    'Tuesday': 'Weekday',
    'Wednesday': 'Weekday',
    'Thursday': 'Weekday',
    'Friday': 'Weekday',
    'Saturday': 'Weekend',
    'Sunday': 'Weekend'
}
df_sample['Day_Type'] = df_sample['DayOfWeek'].map(weekday_map)
# Aggregate crime counts
day_type_counts = df_sample['Day_Type'].value_counts()
day_type_share = (day_type_counts/total_crimes)*100
day_type_counts, day_type_share.round(2)


# In[ ]:


peak_day=df_sample['DayOfWeek'].value_counts().idxmax()
print("Peak day is:", peak_day)
peak_day_count=df_sample['DayOfWeek'].value_counts().max()
peak_day_share=(peak_day_count/total_crimes)*100
print(f"Peak day share is: {peak_day_share:.2f}")


# Friday is the day of the week with the highest crime incidence, accounting for about 16% of weekly crimes. This is probably due to the increased mobility during the end of the week or organized events for the weekend.
# * Strengthen police presence on Fridays, in particular during late afternoon-evening hours.

# In[ ]:


# Filter peak day (Friday)
friday_df = df_sample[df_sample['DayOfWeek'] == 'Friday']

# Crime category distribution on Friday
friday_category_share = (friday_df['Category'].value_counts(normalize=True) * 100)

# Overall crime category distribution
overall_category_share = (df_sample['Category'].value_counts(normalize=True) * 100)

# Combine for comparison
category_comparison = pd.concat([overall_category_share, friday_category_share], axis=1, keys=['Overall(%)', 'Friday(%)']).dropna()

category_comparison.sort_values('Friday(%)', ascending=False).head(10)


# In[ ]:


# Comparison datasets for plotting
category_comparison_reset = category_comparison.reset_index()
category_comparison_reset.rename(columns={'index': 'Category'}, inplace=True)

top_categories= (
    category_comparison_reset.
    sort_values('Friday(%)', ascending=False)
    .head(8)) # keep top 8 categories

fig= px.bar(top_categories, x='Category', y=['Overall(%)', 'Friday(%)'],
            barmode='group',
            title='Crime Category Distribution: Overall vs Friday',
            labels={'value': 'Share(%)', 'variable': 'Distribution'})
fig.update_layout(xaxis_title='Crime Category',legend_title_text='')
fig.show()

            
                 


# The graph illustrates a comparison of crime category distributions between Fridays and other days of the week, highlighting that Larceny/Theft crimes occur more frequently on Fridays than on all other days. 

# ### KPI 4: Peak Crime Time Range

# In[ ]:


df_sample['Time'].head()


# In[ ]:


df_sample['Hour'] = pd.to_datetime(df_sample['Time'], format='%H:%M').dt.hour


# In[ ]:


bins=[0,4,8,12,16,20,24]
labels = ['0-4','4-8','8-12','12-16','16-20','20-24']
df_sample['Time_range'] = pd.cut(df_sample['Hour'], bins=bins, labels = labels, right=False)
time_counts= df_sample['Time_range'].value_counts().sort_index()
time_counts


# In[ ]:


peak_time_range=time_counts.idxmax()
print("Time range with most crimes:", peak_time_range)


# In[ ]:


peak_time_range_count = time_counts.max()
print("Number of crimes:", peak_time_range_count)


# In[ ]:


peak_time_share=(peak_time_range_count/total_crimes)*100
print(f"Peak time range share is:{peak_time_share:.2f}")


# In[ ]:


avg_crimes_per_slot = time_counts.mean()
peak_time_deviation = (time_counts.max() - avg_crimes_per_slot) / avg_crimes_per_slot * 100
round(peak_time_deviation, 1)


# The time range between 16:00-20:00 accounts for approximately 24% of daily crimes, showing a strong deviation from the average.
# * Reallocate patrols to concentrate resources between 16:00 and 20:00, reducing coverage during low-incidence hours.

# ### KPI 5: Most critical Police District

# In[ ]:


district_counts = (
    df_sample
    .groupby('PdDistrict')
    .size()
    .reset_index(name='NumCrimes')
    .sort_values('NumCrimes', ascending=False)
)
district_counts


# In[ ]:


top_district = district_counts.iloc[0]
print("Top district is:", top_district['PdDistrict'])
print("Number of crimes:", top_district['NumCrimes'])


# In[ ]:


crimes_by_district = df_sample.groupby('PdDistrict').size()
crime_share = (crimes_by_district/total_crimes)*100
print(crime_share)


# The Southern District records the highest rate of daily crimes and accounts for approximately 20% of total incidents. 
# * Prioritize the Southern District for permanent patrols. 

# In[ ]:


top_6_categories = df_sample['Category'].value_counts().head(6).index.tolist()
print("Top 6 crime categories:", top_6_categories)


# In[ ]:


df_top6 = df_sample[df_sample['Category'].isin(top_6_categories)]


# In[ ]:


district_category_top6 = (
    df_top6
    .groupby(['PdDistrict', 'Category'])
    .size()
    .reset_index(name='NumCrimes')
)
district_category_top6.head()


# In[ ]:


fig = px.bar(
    district_category_top6,
    x='PdDistrict',
    y='NumCrimes',
    color='Category',
    title='Distribution of the 6 most common crimes vs District',
    text='NumCrimes'
)

fig.update_layout(
    xaxis_title='Police District',
    yaxis_title='Number of Crimes',
    legend_title_text='Crime Category'
)

fig.show()


# MISSION AND SOUTHERN present a higher concentration of ASSAULT crimes in respect to the other Districts. 
# CENTRAL, NORTHERN and SOUTHERN show a higher amount of LARCENY/THEFT incidents

# ### KPI 6: Hotspot Address Concentration

# In[ ]:


top_address= df_sample['Address'].value_counts().idxmax()
print("Top Address is:", top_address)
address_counts = df_sample['Address'].value_counts().head(10)


# In[ ]:


top_address_count = address_counts.iloc[0]
top_address_share=(top_address_count/total_crimes)*100
print("Top Address Share is:", top_address_share)


# In[ ]:


top_5_addresses_share = (
    df_sample['Address']
    .value_counts()
    .head(5)
    .sum() / total_crimes * 100
)
round(top_5_addresses_share, 2)


# At the address 800 Block of BRYANT ST, there is a crime concentration of approximately 2.3% of total incidents in San Francisco, while the top 5 addresses represent a significant concentration. 
# * Apply targeted interventions at hotspot locations for reducing crime incidents. 

# In[ ]:


# CRIME PATTERN VISUALIZATION
df_sample['Date'] = pd.to_datetime(df_sample['Date'])
df_sample['DateTime'] = df_sample['Date'] + pd.to_timedelta(df_sample['Hour'], unit='h')

# Creating 'Hour' and 'DayOfWeek' columns
df_sample['Hour'] = df_sample['DateTime'].dt.hour
df_sample['DayOfWeek'] = df_sample['DateTime'].dt.day_name()

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_sample['DayOfWeek'] = pd.Categorical(df_sample['DayOfWeek'], categories=day_order, ordered=True)

districts = df_sample['PdDistrict'].unique()


# In[ ]:


# Line plot for each Distruct
hourly_pattern = df_sample.groupby(['PdDistrict', 'Hour']).size().reset_index(name='NumCrimes')

plt.figure(figsize=(12,6))
for district in districts:
    data = hourly_pattern[hourly_pattern['PdDistrict'] == district]
    sns.lineplot(data=data, x='Hour', y='NumCrimes', marker='o', label=district)

plt.title('Hourly Crime Pattern vs District')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Crimes')
plt.xticks(range(0,24))
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# top 3 hotspot address:
top_address=(df_sample['Address'].value_counts().head(3).index)
top_address


# In[ ]:


hotspot_df=df_sample[df_sample['Address'].isin(top_address)]
# counting of crimes for Address and Category
hotspot_category_counts = (
    hotspot_df
    .groupby(['Address', 'Category'])
    .size()
    .reset_index(name='NumCrimes')
)
hotspot_category_counts.head()

# share for each address
hotspot_category_share = (
    hotspot_category_counts
    .groupby('Address')
    .apply(lambda x: x.assign(SharePct=100 * x['NumCrimes'] / x['NumCrimes'].sum()))
    .reset_index(drop=True)
)

hotspot_category_share.head()


# In[ ]:


fig = px.bar(hotspot_category_share,
    x='Category',
    y='SharePct',
    color='Address',
    barmode='group',
    title='Crime Type Distribution by Hotspot Address',
    labels={'SharePct': 'Share (%)', 'Category': 'Crime Category'}
)

fig.update_layout(xaxis_tickangle=45)
fig.show()


# The analysis shows that hotspot Address do not to have the same crime distribution. In particular, 800 Block of Market ST is dominated by Larceny/Theft and Assault, typically associated with social/commercial activity, while other hotspots (e.g. 900 Block of Market ST) show a higher incidence of Robbery and Drug/Narcotic.
#   800 Block of Bryant ST, which records the highest percentage of crimes in San Francisco presents a predominance of Larceny/Theft along with a relatively higher share of Vandalism compared to other hotspots. 
# These differences suggest the need for differentiated, location-specific intervention strategies between different hotspots. 

# ## 4. Geospatial Analysis

# In[ ]:


# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42

# create map and display it
sanfrancisco_map = folium.Map(location=[latitude, longitude], zoom_start=12)
sanfrancisco_map


# In[ ]:


# creating a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

# loop through the 10000 crimes and add each to the incidents feature group
for lat, lng, in zip(df_sample.Y, df_sample.X):
    incidents.add_child(
        folium.vector_layers.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add incidents to map
sanfrancisco_map.add_child(incidents)


# A better visualization would be to group the markers into different clusters. Each cluster is then represented by the number of crimes in each neighborhood. 

# In[ ]:


from folium import plugins

# clean copy of the map of San Francisco
sanfrancisco_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# creating a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(sanfrancisco_map)

# loop through the dataframe and add each data point to the marker cluster
for lat, lng, label, in zip(df_sample.Y, df_sample.X, df_sample.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map
sanfrancisco_map


# In[ ]:


# extracting the unique crime categories from the dataset and assign a color to each category
# list of unique categories
categories = df_sample['Category'].unique()

# color palette
colors = [
    'red', 'blue', 'green', 'purple', 'orange',
    'darkred', 'cadetblue', 'darkgreen', 'pink', 'gray'
]

# dictionary mapping category → color
color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

color_map


# In[ ]:


# Map with CircleMarkers colored by category
# create the map
sanfrancisco_map = folium.Map(
    location=[latitude, longitude],
    zoom_start=12,
    tiles='CartoDB positron'
)

# add the markers
for lat, lng, category in zip(df_sample.Y, df_sample.X, df_sample.Category):
    folium.CircleMarker(
        location=[lat, lng],
        radius=6,
        color=color_map[category],
        fill=True,
        fill_color=color_map[category],
        fill_opacity=0.7,
        popup=category
    ).add_to(sanfrancisco_map)

# display the map
sanfrancisco_map


# Thi high amount of LARCENY/THEFT crinmes is very well visible in the map with blue color. 

# ## 5. Conclusions
# This analysis highlights clear temporal, spatial and categorical crime patterns in san Francisco. Crime incidents are concentrated in specific time windows (16:00-20:00), Police Districts (SOUTHERN) and hotspot locations (800 Block of BRYANT ST) with LARCENY/THEFT emerging as the dominant crime category.
# Therefore, the results suggest that crimes in San Francisco are not uniformly distributed ehich means that improving the efficiency of police resources is the best way to support crime prevention efforts. 

# ## 6. Final Recommendations
# 
# * Focus 30–40% of patrol resources during the 16:00–20:00 time window
# * Prioritize the Southern District for preventive and permanent policing
# * Address crime hotspots with targeted, location-specific interventions
# * Emphasize prevention of LARCENY/THEFT in commercial and high-traffic areas

# ## 7. Data Quality and Limitation:
# * The presented analysis consider a sample of 10000 measures and no normalization has been done on the overall population (150000).
# * The data is anonymized
# * The socio-economic variables are missing
# 

# In[ ]:




