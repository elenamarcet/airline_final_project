def load_data(yalm_path):
    """
    Lee un archivo de configuración YAML, carga el dataset, realiza varias transformaciones en el dataset y devuelve el DataFrame resultante.

    Args:
        config_path (str): Ruta al archivo de configuración YAML.
    
    Returns:
        pandas.DataFrame: DataFrame procesado con las transformaciones aplicadas.
    """

    import pandas as pd
    import yaml

    #importamos el dataset desde el archivo yaml
    with open ('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

    #leemos el dataset
    df = pd.read_csv(config['data']['df'])

    #Cambio nombre de columnas 
    df = df.rename(columns={'satisfaction':'Satisfaction', 'Customer Type':'Customer_type', 'Type of Travel':'Type_of_travel', 'Flight Distance':'Flight_distance', 'Seat comfort':'Seat_comfort', 'Departure/Arrival time convenient':'Departure/Arrival_time', 'Food and drink':'Food_and_drink', 'Gate location':'Gate_location', 'Inflight wifi service':'Inflight_wifi_service', 'Inflight entertainment':'Inflight_entertaiment', 'Online support':'Online_support', 'Ease of Online booking':'Ease_of_online_booking', 'On-board service':'Onboard_service', 'Leg room service':'Leg_room_service', 'Baggage handling':'Baggage_handling', 'Checkin service':'Checkin_service', 'Online boarding': 'Online_boarding', 'Departure Delay in Minutes':'Departure_delay_in_minutes', 'Arrival Delay in Minutes':'Arrival_delay_in_minutes'})

    #elimino las filas con los valores nulos en arrival delay minutes
    df = df.dropna(subset=['Arrival_delay_in_minutes'])

    return df