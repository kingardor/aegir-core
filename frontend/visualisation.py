import streamlit as st
import cv2
from streamlit.server.server import Server
from PIL import Image
import session_state as SessionState
import numpy as np
import pandas as pd
import pydeck as pdk
import time
import requests
import base64
import json

st.set_page_config(page_title="Kludge Inc.", page_icon="🔱")
'''
# Aegir 🔱

# Live Visualisation of Time-Series Prediction of Debris Movement
'''
def conv_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def force_rerun():
    session_infos = Server.get_current()._session_info_by_id.values()
    for session_info in session_infos:
        this_session = session_info.session
    this_session.request_rerun()

def hide_streamlit_widgets():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def img2b64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return(jpg_as_text)

def visualize_map(df_runtime,stop):
    counter = 0
    map_obz = st.empty()
    year_met = st.empty()
    map_obz.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        initial_view_state=pdk.ViewState(
            latitude=33.1864,
            longitude=-128.7975,
            zoom=0,
            pitch=50,
        ),
        layers = [
            pdk.Layer(
                'HeatmapLayer',
                data=df_runtime,
                get_position='[lon, lat]',
                radius=300,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df_runtime,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=30,
            ),
        ],
    ))
    return

def visualize_2d_map(dataframe,stop):
    counter = 0
    map_obz = st.empty()
    year_met = st.empty()
    _stop = stop.button("stop visualisations")
    for i in range(len(dataframe)):
        if _stop:
            break
        df_runtime = dataframe.iloc[:counter,:]
        time.sleep(0.25)
        map_obz.map(df_runtime)
        counter += 1
    return

def main():

    start_visualisation = st.empty()
    stop_visualisation = st.empty()
    visualisations_state = SessionState.get(enabled = True, enabled_visualisations = True, run = False)

    if visualisations_state is not None:
        if not visualisations_state.run:
            start = start_visualisation.button("Visualise")
            visualisations_state.start = start

        if visualisations_state.run:
            visualisations_state.enabled = True
            visualisations_state.run = False
            df = pd.read_csv('latlong-filtered-timeseries.csv', usecols= ['lat','lon'])
            visualize_map(df,stop_visualisation)
            visualize_2d_map(df,stop_visualisation)
        else:
            visualisations_state.run = True

main()
'''

'''