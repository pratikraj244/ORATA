import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import whisper
import tempfile
import os
from PIL import Image
import base64
from io import BytesIO
import speech_recognition as sr
import spacy
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_extras.metric_cards import style_metric_cards
st.set_page_config(
    page_title="ORATA",
    layout = "wide"
)
def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str
def pil1_to_base64(img1):
    buffer = BytesIO()
    img1.save(buffer, format="PNG")
    img1_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img1_str
img = Image.open(r"orata.png")
img_base64 = pil_to_base64(img)

img1 = Image.open(r"back_image1.png")
img1_base64 = pil1_to_base64(img1)

img2 = Image.open(r"back_image2.png")
img2_base64 = pil1_to_base64(img2)


st.markdown("""<style>[data-testid="stSidebar"] {
        background-color: #2e2e2e !important;
    }</style>""",unsafe_allow_html=True)
st.markdown("""
<style>
    /* Main sidebar container */
    section[data-testid="stSidebar"] > div {
        padding: 0rem !important;
    }
    
    /* Option menu container */
    .st-emotion-cache-1cypcdb {
        width: 200% !important;
        padding: 0 !important;
        margin: 0 !important;
        background-color: #2b2b2b;
        border-radius: 0px;
    }
    
    /* Menu items */
    .st-emotion-cache-1wbqy5l {
        width: 100% !important;
        margin: 0 !important;
        text-align: left !important;
        padding-left: 10px !important;
        padding: 15px 15px !important;
        box-sizing: border-box !important;
    }
    
    /* Selected menu item */
    .st-emotion-cache-ffhzg2 {
        width: 100% !important;
        margin: 0 !important;
        text-align: left !important;
        padding-left: 10px !important
    }
    
    /* Remove gaps between items */
    ul[role="menu"] {
        gap: 0 !important;
    }
</style>
""", unsafe_allow_html=True)
with st.sidebar:
   st.markdown("""
<style>
    .hero-image{
        max-width: 80%;
        margin-left: 26px;
        margin-top: -20px   
            }
""",unsafe_allow_html=True)
   st.markdown(f"""
<div class="hero-image">
            <img src="data:image/png;base64,{img_base64}" />
        </div>
""",unsafe_allow_html=True)
   selected = option_menu(
        menu_title=None,
        options = ["ANALYTICS","REFINE TALK","ABOUT ME"],
        icons=["bar-chart", "mic", "info-circle"],
        menu_icon="cast",
        default_index=1,
        styles={
            "container": {
                "padding": "0",
                "margin": "0",
                "width": "100%",
                "background-color": "#2e2e2e",
                "border-radius": "0px"
            },
            "nav-link": {
                "width": "100%",
                "margin": "0",
                "margin-left": "5px",
                "text-align": "left",
                "padding": "16px 16px",
                "font-size": "16px",
                "text-align": "left",
                "color": "white",
            },
            "nav-link-selected": {
                "width": "100%",
                "margin": "0",
                "padding": "16px 16px",
                "background-color": "transparent",
                "color": "#00CDBD",
                "font-weight": "bold",
                "border-radius": "0",
            },
            "icon": {
                "color": "white",
                "font-size": "20px",
            }
        }
   )
   st.markdown("""
        <style>
            div[data-testid="stFileUploader"] > label {
                color: white;
                font-weight: 1600;
                font-size: 24px !important;
               display: flex !important
            }

            

            div[data-testid="stFileUploader"] button {
                background-color: #00CDBD;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 8px;
                border: none;
                font-size: 14px;
                margin-top: -10px;
            }

            div[data-testid="stFileUploader"] button:hover {
                background-color: #00b2a8;
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)
   st.markdown("""
               <style>
                .description-text1{
                 color: #00b9a3;
                 font-size: 40px;
                 font-weight: 1000; 
                 margin-top: 30px;    
               }
               </style>
               """,unsafe_allow_html=True)
   st.markdown("""
               <style>
                .description-text2{
                 color: #20ccaa;
                 font-size: 40px;
                 font-weight: 1000; 
                 margin-top: 0px;    
               }
               </style>
               """,unsafe_allow_html=True)
   st.markdown("""
               <style>
                .description-text3{
                 color: #18d596;
                 font-size: 40px;
                 font-weight: 1000; 
                 margin-top: 0px;
                 font-family: 'Poppins', sans-serif;    
               }
               </style>
               """,unsafe_allow_html=True)
   st.markdown("""
               <style>
                .description-text4{
                 font-size: 18px;
                 font-weight: 400; 
                 margin-top: 15px;
       
               }
               </style>
               """,unsafe_allow_html=True)
   st.markdown("""
                    <style>
                    .grey-box {
                        background-color: #2e2e2e; /* Dark grey */
                        padding: 20px;
                        border-radius: 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)
   d = st.file_uploader(" ", type=["wav"])
if selected == "REFINE TALK":
   st.markdown("""
              <style>
                 .hero-image1{
                  max-width: 200%;
                  margin-left: 0px;
                  margin-top: -20px
               }              
               """,unsafe_allow_html=True)
   st.markdown(f"""
<div class="hero-image1">
            <img src="data:image/png;base64,{img1_base64}" />
        </div>
""",unsafe_allow_html=True)
   #model = whisper.load_model("medium")
   # Save uploaded file to a temporary location
   if d is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(d.read())
        tmp_path = tmp.name

    try:
        st.markdown("<div class='description-text1'> CORRECTED SPEECH </div class>",unsafe_allow_html=True)
        model = whisper.load_model("medium")  # or "medium" as needed
        result = model.transcribe(tmp_path)
        #st.write("**Transcription:**")
        st.write(result["text"])
    except Exception as e:
        st.error(f"❌ Error: {e}")
elif selected == "ANALYTICS":
   #st.markdown("<div class='description-text2'> ANALYTICS </div class>",unsafe_allow_html=True)
   st.markdown("""
              <style>
                 .hero-image2{
                  max-width: 200%;
                  margin-left: 0px;
                  margin-top: -20px
               }              
               """,unsafe_allow_html=True)
   st.markdown(f"""
<div class="hero-image2">
            <img src="data:image/png;base64,{img2_base64}" />
        </div>
""",unsafe_allow_html=True)
   r = sr.Recognizer()
   if d is not None:
      with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as tmp:
         tmp.write(d.read())
         tmp_path = tmp.name
      try:
         with sr.AudioFile(tmp_path) as m:
            r.adjust_for_ambient_noise(m)
            aud = r.record(m)
            text = r.recognize_google(aud,language="eng-in")
            #st.write(text)
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
            col7, col8, col9 = st.columns(3)
            nlp = spacy.load("en_core_web_sm")
            filler = [
                "um", "uh", "like", "you know", "so", "actually",
                "basically", "literally", "i mean", "you see",
                "well", "okay", "right", "hmm", "er", "ah"
            ]
            text11 = text.lower()
            text11_counts = 0
            count = 0
            for i in filler:
                if i in text11:
                    count += 1
                    print(i)
            doc3 = nlp(text11)
            for i in doc3:
                if not i.is_punct:
                    text11_counts += 1
            fill = count/text11_counts
            d = pd.DataFrame({"tokens":["non fillers","fillers"],"count":[(text11_counts-count),count]}, index=[0,1])
            col = ["#00b9a3" if i == "non fillers" else "rgba(0,0,0,0)" for i in d["tokens"]]
            fig = go.Figure(go.Pie(labels=d["tokens"],values=d["count"],marker_colors=col,hole=0.5))
            fig.update_layout(width=260,height=210,margin=dict(t=20, b=0, l=0, r=0),legend=dict(
            x=1,          # move legend horizontally (0 = left, 1 = right)
            y=1,          # move legend vertically (0 = bottom, 1 = top)
            xanchor='left', # anchor legend position
            yanchor='middle',
            orientation='v',
            ))
            # vocab richness
            words = [i.text for i in doc3 if not i.is_punct and not i.is_space]
            w1 = len(words)
            words2 = len(set(words))
            voc = (words2/w1)
            d3 = pd.DataFrame({"tokens":["unique","non unique"],"count":[words2,(w1-words2)]},index=[0,1])
            col = ["#57ffa4" if i == "unique" else "rgba(0,0,0,0)" for i in d3["tokens"]]
            fig1 = go.Figure(go.Pie(labels=d3["tokens"],values=d3["count"],marker_colors=col,hole=0.5))
            fig1.update_layout(
                font = {'color': "white", 'family': "Arial"},
                width=260,height=210,margin=dict(t=20, b=0, l=0, r=0),legend=dict(
            x=1,          # move legend horizontally (0 = left, 1 = right)
            y=1,          # move legend vertically (0 = bottom, 1 = top)
            xanchor='left', # anchor legend position
            yanchor='middle',
            orientation='v',

            ))
            from textblob import TextBlob
            blob = TextBlob(text)
            sp = (blob.sentiment.polarity)  # range is between -1 to +1 , -1 = negative, +1 = positive
            sub = (blob.sentiment.subjectivity) # range is between 0 to 1, 0 = factual(just facts), 1 = (personal feelings, emotions)
            fig2 = go.Figure(go.Indicator(mode="gauge+number",value=sub,number={'font': {'size': 45}},gauge={'axis':{'range':[0,1],
            'tickvals': [0, 1],
            'ticktext': [
                '<span style="font-size:30px;">0</span>',
                '<span style="font-size:30px;">1</span>'
            ]},'bar':{'color':'#00b9a3',"thickness":1}}))
            fig2.update_layout(
                font = {'color': "white", 'family': "Arial"},
                width=260,height=210,margin=dict(t=20, b=0, l=0, r=0),legend=dict(
            x=1,          # move legend horizontally (0 = left, 1 = right)
            y=1,          # move legend vertically (0 = bottom, 1 = top)
            xanchor='left', # anchor legend position
            yanchor='middle',
            orientation='v',
            ))
            fig4 = go.Figure(go.Indicator(mode="gauge+number",value=sp,number={'font': {'size': 45}},gauge={'axis':{'range':[-1,1],'tickvals': [-1,0, 1],
            'ticktext': [
                '<span style="font-size:0px;">-1</span>',
                '<span style="font-size:0px;">0</span>',
                '<span style="font-size:0px;">1</span>'
            ]},'bar':{'color':"#20ccaa","thickness":1}}))
            fig4.update_layout(
                font = {'color': "white", 'family': "Arial"},
                width=260,height=210,margin=dict(t=20, b=0, l=0, r=0),legend=dict(
            x=1,          # move legend horizontally (0 = left, 1 = right)
            y=1,          # move legend vertically (0 = bottom, 1 = top)
            xanchor='left', # anchor legend position
            yanchor='middle',
            orientation='v',

            )

            )
            sn = SentimentIntensityAnalyzer()
            score = sn.polarity_scores(text)
            param =[]
            val = []
            x3 = 0
            for key, value in score.items():
                param.append(key)
                val.append(value)
                if key == "compound":
                    x3 = value
            emo = pd.DataFrame({"param":param,"val":val})
            emo1 = emo.copy()
            emo1.drop(3,inplace=True)
            per = x3*100  # compound calculation
            fig6 = go.Figure(go.Indicator(mode="gauge+number",value=per,number={'font': {'size': 45}},gauge={'axis':{'range':[0,100],
            'tickvals': [0, 100],
            'ticktext': [
                '<span style="font-size:30px;">0</span>',
                '<span style="font-size:30px;">1</span>'
            ]},'bar':{'color':"#59e5e8","thickness":1}}))
            fig6.update_layout(
                font = {'color': "white", 'family': "Arial"},
                width=260,height=210,margin=dict(t=20, b=0, l=0, r=0),legend=dict(
            x=1,          # move legend horizontally (0 = left, 1 = right)
            y=1,          # move legend vertically (0 = bottom, 1 = top)
            xanchor='left', # anchor legend position
            yanchor='middle',
            orientation='v',

            )

            )

            fig5 = px.bar(emo1,x="param",y="val",color_discrete_sequence=["#23e0aa"])
            fig5.update_layout(
                font = {'color': "white", 'family': "Arial"},
                width=260,height=210,margin=dict(t=20, b=0, l=0, r=0),legend=dict(
            x=1,          # move legend horizontally (0 = left, 1 = right)
            y=1,          # move legend vertically (0 = bottom, 1 = top)
            xanchor='left', # anchor legend position
            yanchor='middle',
            orientation='v',

            )

            )
            #fig4.show() # sentiment polarity
            with col4:
                st.subheader("Fillers Analysis")
                st.plotly_chart(fig,use_container_width=False)
                st.write("This graph calculates the filler words presence. Less filler words in the speech indicates speech. Therefore filler words such as 'umm', 'uh!', 'you see', 'er', etc should be avoided in order to make the speech fluent. ")
            with col5:
                st.subheader("Vocab Richness")
                st.plotly_chart(fig1,use_container_width=False)
                st.write("Vocab Richness calculated by presence of unique words in the speech. Try to bring vocabulary richness in your speech, but using simple words efficiently would be more than enough.")
            
            with col6:
                st.subheader("Subjective tone")
                st.plotly_chart(fig2,use_container_width=False)
                st.write("This score calculates whether the speech is inclined towards factual or emotional conversation. Towards 0, the speech indicates to be more factual and towards 1, to be more of emotional or personal.")

            with col7:
                st.subheader("Feeling score")
                st.plotly_chart(fig4,use_container_width=False)
                st.write("This score calculates whether the tone of speech is positive, negative or neutral, just like sentiment analysis. Towards -1, the speech indicates to be more negative and towards 1, to be more optimistic.")
            with col8:
                st.subheader("Sentiment Analyzer")
                st.plotly_chart(fig5, use_container_width=False)
                st.write("It analyses the wordings which are postive, negative or neutral. In professional interviews, ideal speech consists of neutral wordings with bit of positive ones.")
            with col9:
                st.subheader("Compound Score")
                st.plotly_chart(fig6,use_container_width=False)
                st.write("Compound score checks the overall sentiment of the speech, whether its impact is postive or negative. Higher compound score means your speech is providing more positive impact. ")
  
      except:
           st.write("AUDIO NOT RECIEVED")
if selected == "ABOUT ME":
    st.markdown("<div class='description-text3'> Hi. It's me ORATA </div class>",unsafe_allow_html=True)
    st.markdown("<div class='description-text4'> Your AI-powered speech coach — a smart companion dedicated to helping you improve your speaking skills in a practical, results-driven way. I will listen to your speech in audio form, identify areas for improvement, and deliver a refined, polished version that is clear, confident, and suitable for professional settings such as interviews, business meetings, presentations, and conferences. Beyond just correction, I will provide in-depth, data-driven insights and analytics on your speech — including pace, clarity, filler word usage, sentiment tone, and vocabulary richness — so you can understand exactly how you communicate and where you can enhance your delivery. These insights will serve as an actionable roadmap, empowering you to practice effectively, refine your style, and communicate with greater impact. A good speech connects people, and brings clarity, that helps to share your thoughts and ideas to other person in front of you. </div class>",unsafe_allow_html=True)
