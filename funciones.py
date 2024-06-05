def load_data():
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
    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    #leemos el dataset
    df = pd.read_csv(config['data']['df'])

    #Cambio nombre de columnas 
    df = df.rename(columns={'satisfaction':'Satisfaction', 'Customer Type':'Customer_type', 'Type of Travel':'Type_of_travel', 'Flight Distance':'Flight_distance', 'Seat comfort':'Seat_comfort', 'Departure/Arrival time convenient':'Departure/Arrival_time', 'Food and drink':'Food_and_drink', 'Gate location':'Gate_location', 'Inflight wifi service':'Inflight_wifi_service', 'Inflight entertainment':'Inflight_entertaiment', 'Online support':'Online_support', 'Ease of Online booking':'Ease_of_online_booking', 'On-board service':'Onboard_service', 'Leg room service':'Leg_room_service', 'Baggage handling':'Baggage_handling', 'Checkin service':'Checkin_service', 'Online boarding': 'Online_boarding', 'Departure Delay in Minutes':'Departure_delay_in_minutes', 'Arrival Delay in Minutes':'Arrival_delay_in_minutes'})

    #elimino las filas con los valores nulos en arrival delay minutes
    df = df.dropna(subset=['Arrival_delay_in_minutes'])

    return df

def graph_distribution_satisfaction(df):
    """
    Calcula el porcentaje de clientes que están satisfechos y no del dataset, y crea un gráfico circular para visualizar esta distribución.

    Args: 
        df(pandas.DataFrame): DataFrame que contiene la columna 'Satisfaction' con la información de los clientes que estan satifechos y no. 
    
    Returns:
        plotly.graph_objects. Figure: Figura del gráfico circular mostrando la distribución de clientes satisfechos y insatisfechos
    """
    import pandas as pd
    import plotly.express as px

    #calculamos el porcentaje de clientes satifechos y insatisfechos
    satisfied_percent = (df['Satisfaction'].value_counts(normalize=True)*100).reset_index()

    #creamos el gráfico circular para ver la distribución de clientes satisfechos y insatisfechos
    fig = px.pie(satisfied_percent, values='proportion', names='Satisfaction', title='Distribution of satisfied and dissatisfied customers')

    #mostramos el gráfico
    fig.show()

def graph_satisfaction(df):
    """
    Calcula la cantidad de clientes que están satisfechos y insatisfechos del dataset, y crea un gráfico de barras.

    Args: 
        df(pandas.DataFrame): DataFrame que contiene la columna 'Satisfaction' con la información de los clientes que estan satifechos y no. 

    Returns:
        seaborn.graph_objects. Figure: Figura del gráfico de barras mostrando los clientes satisfechos y insatisfechos
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    #creamos el grafico de clientes satisfechos y insatisfechos
    sns.countplot(x='Satisfaction', data=df)
    plt.title('Satisfaction')

    #mostramos gráfico
    plt.show()


def graph_satisfaction_customer_type(df):
    """
    Calcula la cantidad de clientes que están satisfechos y insatisfechos segun el tipo de clientes que son, y crea un gráfico de barras.

    Args: 
        df(pandas.DataFrame): DataFrame que contiene la columna 'Satisfaction' y 'Customer_type' con la información de los clientes.

    Returns:
        seaborn.graph_objects. Figure: Figura del gráfico de barras mostrando los clientes satisfechos y insatisfechos segun el tipo
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    #creamos el grafico de clientes satisfechos y insatisfechos segun el tipo de clientes
    sns.countplot(x='Satisfaction', hue='Customer_type', data=df)
    plt.title('Satisfaction by Customer Type')

    #mostramos gráfico
    plt.show()

def graph_satisfaction_travel_type(df):
    """
    Calcula la cantidad de clientes que están satisfechos y insatisfechos segun el tipo de viaje, y crea un gráfico de barras.

    Args: 
        df(pandas.DataFrame): DataFrame que contiene la columna 'Satisfaction' y 'Type_of_travel' con la información de los clientes.

    Returns:
        seaborn.graph_objects. Figure: Figura del gráfico de barras mostrando los clientes satisfechos y insatisfechos segun el tipo de viaje
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    #creamos el grafico de clientes satisfechos y insatisfechos segun el tipo de clientes
    sns.countplot(x='Satisfaction', hue='Type_of_travel', data=df)
    plt.title('Satisfaction by Type of Travel')

    #mostramos gráfico
    plt.show()

def graph_satisfaction_class(df):
    """
    Calcula la cantidad de clientes que están satisfechos y insatisfechos segun la clase que viajan, y crea un gráfico de barras.

    Args: 
        df(pandas.DataFrame): DataFrame que contiene la columna 'Satisfaction' y 'Class' con la información de los clientes.

    Returns:
        seaborn.graph_objects. Figure: Figura del gráfico de barras mostrando los clientes satisfechos y insatisfechos segun la clase que viajan
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    #creamos el grafico de clientes satisfechos y insatisfechos segun la clase en que viajan
    sns.countplot(x='Satisfaction', hue='Class', data=df)
    plt.title('Satisfaction by Class')

    #mostramos gráfico
    plt.show()

def graph_age_distribution(df):
    """
    Crea un histograma con la distribución de las edades de los clientes.

    Args: 
        df(pandas.DataFrame): DataFrame que contiene la columna 'Age'.

    Returns:
        seaborn.graph_objects. Figure: Figura del histograma mostrando la distribución de las edades de los clientes
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    #creamos el grafico de distribución de edades
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Distribución de Edades')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')

    #mostramos gráfico
    plt.show()

def graph_numeric_var_density(df):
    """
    Crea una figura con subgráficos para mostrar la densidad de las variables numéricas en el DataFrame.

    Args: 
        df(pandas.DataFrame): DataFrame que contiene las variables numéricas a analizar
    
    Returns:
        matplotlib.figure.Figure: Figura que contiene los subgráficos con la distribución de densidad de las variables numéricas.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    #comprobamos la distribución de todas las variables numéricas
    numeric_var = ['Age', 'Flight_distance', 'Seat_comfort', 'Departure/Arrival_time', 'Food_and_drink', 'Gate_location', 'Inflight_wifi_service', 'Inflight_entertaiment', 'Online_support', 'Ease_of_online_booking', 'Onboard_service', 'Leg_room_service', 'Baggage_handling', 'Checkin_service', 'Cleanliness', 'Online_boarding', 'Departure_delay_in_minutes', 'Arrival_delay_in_minutes']

    ax = df.hist(figsize=(15, 20), bins=60, xlabelsize=10, ylabelsize=10)
    for axis in ax.flatten():
        axis.ticklabel_format(style='plain', axis='x')
    
    #mostramos gráficos
    plt.show()

def graph_correlation_heatmap(df, numeric_var):
    """
    Crea un gráfico de correlación de las variables numéricas del dataframe.

    Arg: 
        df (pandas.DataFrame): El dataframe que contiene las variables numéricas.
        numeric_var (list): Lista de nombres de las variables numéricas a incluir en el gráfico de correlación.
    
    Returns:
    None
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    #Calculamos la matriz de correlación
    corr=np.abs(df[numeric_var].corr())

    #Creamos la máscara para la representación triangular
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    #Configuramos el gráfico de matplotlib
    f, ax = plt.subplots(figsize=(10, 10))

    #Generamos un mapa de colores personalizado
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    #Dibujamos el heatmap con la máscara y la relación de aspecto correcta
    sns.heatmap(corr, mask=mask,  vmax=1,square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=corr)

    #Mostramos el mapa de calor
    plt.show()

def chi_squared_heatmap(df, categoric_var):
    """
    Calcula el chi cuadrado para cuantificar la relación entre las variables categóricas y
    crea un mapa de calor para visualizar los resultados.

    Args:
    df (pandas.DataFrame): El dataframe que contiene las variables categóricas.
    categoric_var(list): Lista de nombres de las variables categóricas.

    Returns:
    None
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import chi2_contingency

    #comprobamos la distribución de todas las variables categoricas
    categoric_var = ['Customer_type', 'Class','Type_of_travel', 'Satisfaction']

    #calculamos chi-square
    results = {}
    for col1 in df[categoric_var]:
        for col2 in df[categoric_var]:
            if col1 != col2:
                contingency_table = pd.crosstab(df[categoric_var][col1], df[categoric_var][col2])
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                results[(col1, col2)] = {'chi2': chi2, 'p': p}
    
    #organizamos los datos en una matriz cuadrada
    chi_squared_matrix = pd.DataFrame.from_dict(results, orient='index').reset_index()

    #creamos un mapa de calor con seaborn
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(chi_squared_matrix.pivot(index='level_0', columns='level_1', values='chi2'),
                      annot=True,
                      cmap='coolwarm',
                      linewidths=0.5,
                      fmt=".2f")
    heatmap.set_title('Chi-Squared correlation map among categoric vars')

    #Mostramos el mapa de calor
    plt.show()

def satisfaction_to_numeric(df):
    """
    Convierte una columna del dataframe de tipo categórico a tipo numérico.

    Args:
    df (pandas.DataFrame): El dataframe que contiene la columna a convertir.
    
    Returns:
    pandas.DataFrame: El dataframe con la columna convertida a tipo numérico.
    """

    df['Satisfaction'] = df['Satisfaction'].map({'dissatisfied': 0, 'satisfied': 1})

    return df


def prediction_model(df_dummies):
    """
    Entrena un modelo de clasificación utilizando un clasificador Gradient Boosting, lo evalúa y muestra las métricas de evaluación.

    Args:
    df_dummies (pandas.DataFrame): El dataframe que contiene las variables dummy.

    Returns:
    tuple: Una tupla que contiene las métricas de evaluación (precision, recall, f1_score).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV

    df_dummies = pd.get_dummies(df, columns = ['Class', 'Customer_type', 'Type_of_travel'])

    #Seleccionamos las variables que emplearemos en el algoritmo y definimos nuestro target en 'Satisfaction'
    features = df_dummies[['Seat_comfort', 'Departure/Arrival_time', 'Food_and_drink', 'Gate_location', 'Inflight_wifi_service', 'Inflight_entertaiment', 'Online_support', 'Ease_of_online_booking', 'Onboard_service', 'Leg_room_service', 'Baggage_handling', 'Checkin_service', 'Cleanliness', 'Online_boarding', 'Departure_delay_in_minutes', 'Arrival_delay_in_minutes']]
    target = df_dummies['Satisfaction']

    #Dividimos los datos en conjuntos de entrenamiento y prueba, donde el 20% de los datos se utiliza para la prueba
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)

    #probamos con el modelo Gradient Boost
    gb= GradientBoostingClassifier(max_depth=20,
                                   n_estimators=2000)
    
    #entrenamos el modelo
    gb.fit(X_train, y_train)

    pred = gb.predict(X_test)

    precision_gradient_boost  = precision_score(y_test, pred, average='macro')
    recall_gradient_boost  = recall_score(y_test, pred, average='macro') 
    f1_gradient_boost  = f1_score(y_test, pred, average='macro')

    #Imprimimos las métricas
    print('Precision:',precision_gradient_boost)
    print('Recall:',recall_gradient_boost)
    print('F1:',f1_gradient_boost)

    #Devolvemos las métricas de evaluación
    return precision_gradient_boost, recall_gradient_boost, f1_gradient_boost