
import streamlit as st  #Importamos Streamlit para crear la web
from funciones import load_data, graph_distribution_satisfaction, satisfaction_to_numeric, prediction_model, graph_featuring_importance #Importamos las funciones del archivo de funciones.py
import pandas as pd
import numpy as np
import plotly.express as px

def main():
    st.title('Airline Satisfaction Dashboard')  #Titulo para la web

    #Cargamos la data del archivo funciones.py
    data = load_data()

    data = satisfaction_to_numeric(data)
    
    gb, X_train, features = prediction_model(data)
    

    fig = graph_featuring_importance(gb, X_train)
    st.plotly_chart(fig)

    #proceso predictivo
    df_to_predict = pd.DataFrame(columns=features.columns)
    

    number_seats = 100
    number_seats = st.slider('How many seats have the Airplane?', min_value=50, max_value=615, value=300, step=50)
    

    #question 1
    percent_of_high_comfort = st.slider('How many percent of seats have high comfort?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_comfort = percent_of_high_comfort/100
    percent_of_low_comfort = 1 - percent_of_high_comfort #3 y 2
    Seat_comfort_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_of_low_comfort)))
    Seat_comfort_temp_3 = list(np.random.choice([1,0], size=int(number_seats*percent_of_high_comfort)))
    df_to_predict['Seat_comfort'] = Seat_comfort_temp_1 + Seat_comfort_temp_3

    #question 2
    percent_of_high_food_payment = st.slider('What percentage of customers purchase food and beverages?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_food_payment = percent_of_high_food_payment/100
    percent_of_medium_food_free = 1 - percent_of_high_food_payment #3 y 2
    Food_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_of_medium_food_free)))
    Food_temp_3 = list(np.random.choice([1,0], size=int(number_seats*percent_of_high_food_payment)))
    df_to_predict['Food_and_drink'] = Food_temp_1 + Food_temp_3

    #question 3
    percent_of_high_gate = st.slider('What percentage of passengers find the access to the exit gate easy?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_gate = percent_of_high_gate/100
    Gate_location_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_of_high_gate)))
    Gate_location_temp_3 = list(np.random.choice([1,0], size=int(number_seats*(1-percent_of_high_gate))))
    df_to_predict['Gate_location'] = Gate_location_temp_1 + Gate_location_temp_3

    #question 4
    percent_of_high_wifi_payment = st.slider('What percentage of passengers opt to pay for in-flight Wi-Fi?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_wifi_payment = percent_of_high_wifi_payment/100
    percent_of_medium_wifi_free = 1 - percent_of_high_wifi_payment #3 y 2
    Wifi_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_of_medium_wifi_free)))
    Wifi_temp_3 = list(np.random.choice([1,0], size=int(number_seats*percent_of_high_wifi_payment)))
    df_to_predict['Inflight_wifi_service'] = Wifi_temp_1 + Wifi_temp_3

    #question 5
    percent_of_high_entertaiment_payment = st.slider('What percentage of passengers opt pay for in-flight entertainment?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_entertaiment_payment = percent_of_high_entertaiment_payment/100
    percent_of_medium_entertaiment_free = 1 - percent_of_high_entertaiment_payment #3 y 2
    Entertaiment_temp_1 = list(np.random.choice([5,4], size=int(round(number_seats*percent_of_medium_entertaiment_free,0))))
    Entertaiment_temp_3 = list(np.random.choice([1,0], size=int(round(number_seats*percent_of_high_entertaiment_payment,0))))
    df_to_predict['Inflight_entertaiment'] = Entertaiment_temp_1 + Entertaiment_temp_3

    #question 6
    percent_of_high_online_support = st.slider('What percentage of passengers have access to online support?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_online_support = percent_of_high_online_support/100
    online_support_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_of_high_online_support)))
    online_support_temp_3 = list(np.random.choice([1,0], size=int(number_seats*(1-percent_of_high_online_support))))
    df_to_predict['Online_support'] = online_support_temp_1 + online_support_temp_3

    #question 7
    percent_of_high_ease_booking_online = st.slider('What percentage of passengers have access to ease booking online?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_ease_booking_online = percent_of_high_ease_booking_online/100
    ease_booking_online_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_of_high_ease_booking_online)))
    ease_booking_online_temp_3 = list(np.random.choice([1,0], size=int(number_seats*(1-percent_of_high_ease_booking_online))))
    df_to_predict['Ease_of_online_booking'] = ease_booking_online_temp_1 + ease_booking_online_temp_3

    #question 8
    binari_options = ['Yes', 'No']
    onboard_service = st.radio('Do customers have access to on-board service?', binari_options) 
    onboard_service = 1 if onboard_service == 'Yes' else 0
    onboard_service_temp_1 = list(np.random.choice([5,4], size=int(number_seats*onboard_service)))
    onboard_service_temp_3 = list(np.random.choice([1,0], size=int(number_seats*(1-onboard_service))))
    df_to_predict['Onboard_service'] = onboard_service_temp_1 + onboard_service_temp_3

    #question 9
    percent_of_high_leg_room_service = st.slider('What percentage of passengers are willing to pay for extra leg room service?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_leg_room_service = percent_of_high_leg_room_service/100
    percent_of_medium_leg_room_service = 1 - percent_of_high_leg_room_service 
    Leg_room_service_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_of_medium_leg_room_service)))
    Leg_room_service_temp_3 = list(np.random.choice([1,0], size=int(number_seats*percent_of_high_leg_room_service)))
    df_to_predict['Leg_room_service'] = Leg_room_service_temp_1 + Leg_room_service_temp_3

    #question 10 
    percent_of_high_baggage_payment = st.slider('What percentage of passengers opt to pay for baggage handling services?', min_value=0, max_value=100, value=30, step=1)
    percent_of_high_baggage_payment = percent_of_high_baggage_payment/100
    percent_of_medium_baggage_free = 1 - percent_of_high_baggage_payment 
    Baggage_handling_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_of_medium_baggage_free)))
    Baggage_handling_temp_3 = list(np.random.choice([1,0], size=int(number_seats*percent_of_high_baggage_payment)))
    df_to_predict['Baggage_handling'] = Baggage_handling_temp_1 + Baggage_handling_temp_3

    #question 11 
    binari_options = ['Yes', 'No']
    checkin_service = st.radio('Do customers have access to check-in service?', binari_options)
    checkin_service = 1 if checkin_service == 'Yes' else 0
    Checkin_service_temp_1 = list(np.random.choice([5,4], size=int(number_seats*checkin_service)))
    Checkin_service_temp_3 = list(np.random.choice([1,0], size=int(number_seats*(1-checkin_service))))
    df_to_predict['Checkin_service'] = Checkin_service_temp_1 + Checkin_service_temp_3

    #question 12
    percent_of_high_clean = st.slider('How many cleanings are conducted on an aircraft for every 10 flights?', min_value=1, max_value=10, value=6, step=1)
    percent_clean_trip = percent_of_high_clean/10
    percent_dirty_trip = (10 - percent_of_high_clean)/10
    Cleanliness_temp_1 = list(np.random.choice([5,4], size=int(number_seats*percent_clean_trip)))
    Cleanliness_temp_3 = list(np.random.choice([1,0], size=int(number_seats*percent_dirty_trip)))
    df_to_predict['Cleanliness'] = Cleanliness_temp_1 + Cleanliness_temp_3

    #question 13
    binari_options = ['Yes', 'No']
    online_boarding = st.radio('Do customers have access to online-boarding service?', binari_options)
    online_boarding = 1 if checkin_service == 'Yes' else 0
    Online_boarding_temp_1 = list(np.random.choice([5,4], size=int(number_seats*online_boarding)))
    Online_boarding_temp_3 = list(np.random.choice([1,0], size=int(number_seats*(1-online_boarding))))
    df_to_predict['Online_boarding'] = Online_boarding_temp_1 + Online_boarding_temp_3

    #
    Departure_delay_in_minutes_options = features.Departure_delay_in_minutes.unique()
    df_to_predict['Departure_delay_in_minutes'] = np.random.choice(Departure_delay_in_minutes_options, size=number_seats)

    Arrival_delay_in_minutes_options = features.Arrival_delay_in_minutes.unique()
    df_to_predict['Arrival_delay_in_minutes'] = np.random.choice(Arrival_delay_in_minutes_options, size=number_seats)

    Departure_options = features['Departure/Arrival_time'].unique()
    df_to_predict['Departure/Arrival_time'] = np.random.choice(Departure_options, size=number_seats)

    
    predict = gb.predict(df_to_predict)

    df_predict = pd.DataFrame(predict).value_counts(normalize=True)
    df_predict_proportion = df_predict.reset_index().rename(columns={0:'Satisfaction_binari'})
    df_predict_proportion['Satisfaction_binari'] = df_predict_proportion['Satisfaction_binari'].astype(str).replace({'0':'Dissatisfied', '1':'Satisfied'})

    fig2 = px.pie(df_predict_proportion, values='proportion', names='Satisfaction_binari', title='Distribution of Satisfied and Dissatisfied Customers')
    st.plotly_chart(fig2)



# Python script entry point
if __name__ == '__main__':
    main()  # Call the main function when the script is executed