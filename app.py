
import streamlit as st  #Importamos Streamlit para crear la web
from funciones import data_import_and_cleaning  #Importamos las funciones del archivo de funciones.py

def main():
    st.title('Airline Satisfaction Dashboard')  #Titulo para la web

    #Cargamos la data del archivo funciones.py
    data = data_import_and_cleaning()