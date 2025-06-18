import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import plotly.express as px
import folium
from streamlit_folium import st_folium, folium_static


import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

# the usual loading process

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
data_path = os.path.join(current_dir, 'data')
out_path = os.path.join(current_dir, 'outputs')     # For the output
# Read the files
df = pd.read_csv(os.path.join(data_path, 'data_predicted.csv'))

# include the raw shap values
shap = pd.read_csv(os.path.join(data_path, 'shap_raw.csv'))


# we're using the original data, so we'll clean it a bit
# we'll also select a subset which doesn't have missing values. Not great, but it'll be a bit easier
# construction_year
df['construction_year'] = df['construction_year'].replace(0, np.nan)
# gps_height
df['gps_height'] = df['gps_height'].apply(lambda x: np.nan if x <= 0 else x)
df['longitude'] = df['longitude'].replace(0, np.nan)
df['latitude'] = df['latitude'].where(df['latitude'] < -0.5, np.nan)

# select a subset
varlist = ('latitude', 'longitude', 'region', 'amount_tsh', 'gps_height', 'population', 'construction_year',
       'extraction_type_class', 'payment', 'water_quality', 'quantity',
       'source', 'waterpoint_type', 'status_group', 'predicted')

df_small = df[np.intersect1d(df.columns, varlist)].dropna()

# taking only the first 5k, at least for now
df = df_small.iloc[0:5000,:]
# create accuracy value
df = df.assign(correct = lambda df: df.status_group == df.predicted)

# there's gotta be an easier way, but this works

f_f = df.apply(lambda x : True if x['status_group'] == 'functional' and x['predicted'] == 'functional' else False, axis=1)
f_fnr = df.apply(lambda x : True if x['status_group'] == 'functional' and x['predicted'] == 'functional needs repair' else False, axis=1)
f_nf = df.apply(lambda x : True if x['status_group'] == 'functional' and x['predicted'] == 'non functional' else False, axis=1)

fnr_f = df.apply(lambda x : True if x['status_group'] == 'functional needs repair' and x['predicted'] == 'functional' else False, axis=1)
fnr_fnr = df.apply(lambda x : True if x['status_group'] == 'functional needs repair' and x['predicted'] == 'functional needs repair' else False, axis=1)
fnr_nf = df.apply(lambda x : True if x['status_group'] == 'functional needs repair' and x['predicted'] == 'non functional' else False, axis=1)

nf_f = df.apply(lambda x : True if x['status_group'] == 'non functional' and x['predicted'] == 'functional' else False, axis=1)
nf_fnr = df.apply(lambda x : True if x['status_group'] == 'non functional' and x['predicted'] == 'functional needs repair' else False, axis=1)
nf_nf = df.apply(lambda x : True if x['status_group'] == 'non functional' and x['predicted'] == 'non functional' else False, axis=1)

df_conf = pd.DataFrame(f_f, columns=['f_f'])
df_conf['f_fnr'] = f_fnr
df_conf['f_nf'] = f_nf
df_conf['fnr_f'] = fnr_f
df_conf['fnr_fnr'] = fnr_fnr
df_conf['fnr_nf'] = fnr_nf
df_conf['nf_f'] = nf_f
df_conf['nf_fnr'] = nf_fnr
df_conf['nf_nf'] =nf_nf

df_map = df[['latitude','longitude','status_group','predicted']].dropna()

df_f = pd.DataFrame({"Region": df['region'], "Functional": f_f, "Functional Needs Repair": f_fnr, "Not Functional": f_nf})
df_fnr = pd.DataFrame({"Region": df['region'], "Functional": fnr_f, "Functional Needs Repair": fnr_fnr, "Not Functional": fnr_nf})
df_nf = pd.DataFrame({"Region": df['region'], "Functional": nf_f, "Functional Needs Repair": nf_fnr, "Not Functional": nf_nf})

# for the heatmap (perhaps not necessary)
df_heat = df_small.groupby('region').sum('correct')


###################


# set up page config, here is where that gets changed
st.set_page_config(
    page_title="Get Pumped Up Dashboard",
    page_icon="🤠",
    layout="wide",
    initial_sidebar_state="expanded")

alt.theme.enable("dark")


###################
###################


# build the sidebar, get the dropdown menues
with st.sidebar:
    st.title('Get Pumped Up Dashboard')

    region_list = list(df.region.unique())
    status_list = list(df.status_group.unique())
    status_list.append('all')
    feature_list = ('extraction_type_class', 'payment', 'water_quality', 'quantity', 'source', 'waterpoint_type')

    selected_region = st.selectbox('Select a region', region_list, index=len(region_list)-1)
    df_selected_region = df[df.region == selected_region]
    df_selected_region_sorted = df_selected_region.sort_values(by="correct", ascending=False)

    selected_status = st.selectbox('Select a status', status_list, index=len(status_list)-1)
    if selected_status == 'all':
        df_use = df
    else:
        df_use = df[df.status_group == selected_status]

    # selected_feature = st.selectbox('Select a feature', feature_list, index=len(status_list)-1)
    # df_selected_feature = df_small[selected_feature]

    # color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    # selected_color_theme = st.selectbox('Select a color theme', color_theme_list)


# now, more layout things:
# number of columns
col = st.columns(1)

# column 0
with col[0]:
    st.markdown('#### Map??')

## now plotlyexpress??
import plotly.express as px

df_temp = df_use[df_use['region'] == selected_region]

fig = px.scatter_map(df_temp, lat="latitude", lon="longitude", zoom=6, color = 'status_group', hover_data = df[['status_group','predicted', 'water_quality','extraction_type_class', 'quantity', 'payment', 'source']], title = 'Pumps per Region, by actual status', subtitle = 'Select a region on the left. Hover for details on each pump')

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(width=800)
st.plotly_chart(fig,use_container_width=True,width=800)



with st.expander('About', expanded=True):
        st.write('''
            - :orange[**Get Pumped Up**] consists of James Donahue, Fatemeh Ebrahimi, Kateryna Ponomarova
            - Our final model used :orange[**XGBoost**] to classify water pumps from Tanzania
            - Special thanks to :orange[**ReDi School**] for this opportunity
            ''')



















