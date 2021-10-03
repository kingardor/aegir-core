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
from comp_vis import ArgusCV

st.set_page_config(page_title="Kludge Inc.", page_icon="🔱")
'''
# Aegir 🔱

# Drone Live Feed
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

def hit_api(img):
    op = [{"img":img2b64(img)}]
    endpoint = "http://127.0.0.1:6969/debrisdet"
    response = requests.post(endpoint,data=json.dumps(str(op)))
    result = eval(response.content)
    return result


def Process_frames(video_source, stop):
    stframe = st.empty()
    _stop = stop.button("stop")
    counter = 0
    while video_source.isOpened():
        ret,frame = video_source.read()
        counter += 1
        if not counter % 5 == 0:
            continue
        if _stop:
            break
        if not ret:
            print("Something is fishy with the stream ...")
        frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hit_api(frm)
        for r in res:
            x0 = r['x0']
            y0 = r['y0']
            x1 = r['x1']
            y1 = r['y1']
            cv2.rectangle(frm,(int(x0),int(y0)),(int(x1),int(y1)),(0,0,255),2)
            crop = frm[int(y0):int(y1),int(x0):int(x1)]
            thermal_image = ArgusCV.create_heatmap(crop,crop,a1=.7,a2=.5)
            frm[int(y0):int(y1),int(x0):int(x1)] = thermal_image
        stframe.image(frm, width=720)
    return


def visualize_map():
    df = pd.read_csv('test5-sorted.csv', usecols= ['lat','lon'])
    st.pydeck_chart(pdk.Deck(
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
                data=df,
                get_position='[lon, lat]',
                radius=300,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=30,
            ),
        ],
    ))

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
    start_button = st.empty()
    stop_button = st.empty()
    state = SessionState.get(upload_key = None, enabled = True, run = False)
    st.text_input("Enter Video Source URL/File", key="videosrc")
    if st.session_state.videosrc is not None:
    # if file_source is not None:
        vid = cv2.VideoCapture(st.session_state.videosrc)

        if not state.run:
            start = start_button.button("start")
            state.start = start

        if state.run:
            state.enabled = True
            state.run = False
            Process_frames(vid,stop_button)
        else:
            state.run = True
            # force_rerun()


main()