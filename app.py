
import streamlit as st  #Importamos Streamlit para crear la web
from funciones import load_data  #Importamos las funciones del archivo de funciones.py

def main():
    st.title('Airline Satisfaction Dashboard')  #Titulo para la web

    #Cargamos la data del archivo funciones.py
    data = load_data()