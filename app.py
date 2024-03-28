import streamlit as st
from prediction import predict_price
import numpy as np
import joblib
import json

st.title("Singapore flat price prediction")

# # Load options from JSON file
with open("town_name.json", "r") as f:
    data_town = json.load(f)

with open("flat_type.json","r") as f1:
    data_flat = json.load(f1)    

with open("storey_no.json","r") as f2:
    data_storey = json.load(f2) 

with open("room_type.json","r") as f3:
    data_room = json.load(f3)

# # Extract keys from the loaded data
options_town = list(data_town.keys())
options_flat = list(data_flat.keys())
options_storey = list(data_storey.keys())
options_room = list(data_room.keys())

col1, col2 = st.columns(2)

with col1:
    town_option = st.selectbox("Select a town :", options_town)
    town_value = data_town[town_option]
    flat_option = st.selectbox("Select a flat type :", options_flat)
    flat_value = data_flat[flat_option]
    storey_option = st.selectbox("Select the storey :", options_storey)
    storey_value = data_storey[storey_option]
with col2:
    room_option = st.selectbox("Select a room type :", options_room)
    room_value = data_room[room_option]
    floor_area = st.number_input("Enter the floor area in sqm") 

#Calling the prediction model
if st.button("Predict price"):
    result = predict_price(np.array([[floor_area,	town_value,	room_value,	storey_value,flat_value	]]))
    st.write("Predicted resale price is :", result[0])
