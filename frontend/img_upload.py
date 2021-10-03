import streamlit as st
import cv2
from streamlit.server.server import Server
from PIL import Image
import session_state as SessionState
import numpy as np
import pandas as pd
import time
import requests
import base64
import json
import tempfile
from random import randint
from comp_vis import ArgusCV

st.set_page_config(page_title="Kludge Inc.", page_icon="🔱")
'''
# Aegir 🔱

# Upload your image here to let community know !!
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

def Process_frames(image):
    stframe = st.empty()
    frm = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

def main():
    upload = st.empty()
    start_button = st.empty()
    state = SessionState.get(upload_key = None, enabled = True, run = False)
    f = upload.file_uploader('Upload Image', key = state.upload_key)
    if f is not None:
        tfile  = tempfile.NamedTemporaryFile(delete = True)
        tfile.write(f.read())
        upload.empty()
        img = tfile.name

        if not state.run:
            start = start_button.button("start")
            state.start = start

        if state.start:
            start_button.empty()
            state.enabled = False
            if state.run:
                tfile.close()
                f.close()
                state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = True
                state.run = False
                Process_frames(img)
            else:
                state.run = True
                # trigger_rerun()
main()
