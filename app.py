import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud 
import os

# Verificar si el archivo se está ejecutando en el entorno de Streamlit o localmente
if os.path.exists('/mnt/data/online_shoppers_intention.csv'):
    file_path = '/mnt/data/online_shoppers_intention.csv'  # Ruta en Streamlit
else:
    file_path = 'C:\\Users\\julyj\\OneDrive\\Desktop\\IRONHACK\\Proyecto Final\\online_shoppers_intention.csv'  # Ruta local

# Cargar el archivo CSV
data = pd.read_csv(file_path)


st.set_page_config(page_title="WillBuyAI", layout="centered")

# CSS personalizado para el fondo y el estilo de la tarjeta especial
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffe6f2;
    }
    .highlighted {
        border: 2px solid gold;
        border-radius: 10px;
        background-color: #fff4e6;
        padding: 10px;
        box-shadow: 0px 0px 15px 5px rgba(255, 215, 0, 0.6);
        text-align: center;
    }
    .highlighted h3 {
        color: #d34f4f;
    }
    .message-card {
        border: 2px solid #ff6666;
        border-radius: 10px;
        background-color: #fffafa;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0px 0px 10px 3px rgba(255, 102, 102, 0.5);
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
    }
    .feather {
        font-size: 100px;
        color: rgba(255, 165, 0, 0.6);
        text-align: center;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    .red-button {
        background-color: #ff4d4d;
        border: none;
        border-radius: 50%;
        padding: 15px 30px;
        color: white;
        font-size: 20px;
        cursor: pointer;
        box-shadow: 0px 0px 10px 3px rgba(255, 77, 77, 0.5);
    }
    .gift-button {
        background-color: #ff4d4d;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        color: white;
        font-size: 20px;
        cursor: pointer;
        box-shadow: 0px 0px 10px 3px rgba(255, 77, 77, 0.5);
        text-align: center;
    }
    .gift-button:hover {
        background-color: #ff6666;
    }
    }
    .description-box {
        border: 2px solid #666;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
        margin: 20px 0;
        font-family: Arial, sans-serif;
    }
    .description-box h3 {
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

variables = {
    "Administrative": "📋 Number of administrative pages visited by the user.",
    "AdministrativeDuration": "⏱️ Total time spent on administrative pages.",
    "Informational": "ℹ️ Number of informational pages visited.",
    "InformationalDuration": "⏳ Total time spent on informational pages.",
    "ProductRelated": "🛍️ Number of product-related pages visited.",
    "ProductRelatedDuration": "⌛ Total time spent on product-related pages.",
    "BounceRates": "↩️ Bounce rate of the user on the pages.",
    "ExitRates": "🚪 Exit rate on the pages visited.",
    "PageValues": "💰 Value of pages based on conversions.",
    "SpecialDay": "🎉 Proximity to special days like Black Friday.",
    "Month": "📅 Month when the visit occurred.",
    "OperatingSystem": "💻 User's operating system.",
    "Browser": "🌐 Browser used.",
    "Region": "🗺️ Geographic region of the user.",
    "TrafficType": "🚦 Type of traffic that brought the user to the site.",
    "VisitorType": "🙋‍♂️ Type of visitor (New/Returning).",
    "Weekend": "📆 Whether the visit occurred during the weekend.",
    "Revenue": "💵 Whether the user made a purchase (target)."
}

# Sidebar
st.sidebar.image("C:\\Users\\julyj\\OneDrive\\Desktop\\IRONHACK\\Proyecto Final\\imagenbuyia.png", width=170)
st.sidebar.title("Sections")
opcion = st.sidebar.radio("Select an Option:", ["Intro", "Data", "EDA Univariate", "EDA Bivariate", "Machine Learning", "Greetings"])

if opcion == "Intro":
    # Mostrar la imagen primero
    st.image(r"C:\\Users\\julyj\\OneDrive\\Desktop\\IRONHACK\\Proyecto Final\\imagenbuyia.png", use_column_width=True)

    # Añadir un pequeño espacio debajo de la imagen
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # Título principal debajo de la imagen
    st.title("🔮 WillBuyAI")

    # Introducción en inglés
    st.markdown("""
    ## 🪄 Welcome to **WillBuyAI** 

    In the dynamic world of e-commerce, understanding buyers' intentions is crucial. 
    **WillBuyAI** is a powerful Machine Learning tool designed to predict whether a user 
    will make a purchase based on their online behavior.

    ### ✨ What do we do at **WillBuyAI**?
    1. 💼 **We analyze user behavior**: From the time they spend on different pages to the actions they take.
    2. 🪄 **We predict purchase intentions**: Using advanced machine learning models, we classify users as buyers or non-buyers.
    3. 🌈 **We optimize strategies**: Helping businesses better understand their customers and improve conversion rates.

    ### 💡 What will you find here?
    - 📊 **Intuitive visualizations**: Understand the data with interactive graphs.
    - 📥 **Real-time predictions**: Upload your data and watch the model in action.
    - 🔍 **Key insights**: Discover patterns that drive decision-making.

    Get ready to explore the minds of buyers with **WillBuyAI** and take your data analysis to the next level.
    """)
elif opcion == "Data":
    st.title("📊 A little bit of Data...")

    # Añadir espacio debajo del título
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # Crear la nube de palabras
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Reds",
        relative_scaling=0.5
    ).generate_from_frequencies({
        **{var: 1 for var in variables.keys()},
        "Revenue": 5
    })

    st.image(wordcloud.to_array(), use_column_width=True)

    # Añadir textos explicativos debajo de la nube de palabras
    st.markdown("""
    **These data reflect the behavior of users in an online store.** Each variable provides valuable insights into how users navigate, interact, and make decisions within the website.
    """)
    
    st.markdown("""
    **Revenue** represents the main goal: understanding whether a user completed a purchase online. This indicator helps identify patterns in customer behavior, aiming to optimize the shopping experience and increase conversions.
    """)

    # Añadir espacio entre la descripción y la tabla
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # Crear tabla como DataFrame
    data = [{"Feature": var, "Description": desc} for var, desc in variables.items()]
    df = pd.DataFrame(data)

    # Mostrar tabla con estilo nativo de Streamlit
    st.table(df)

    # Añadir la bibliografía del dataset
    st.markdown("""
    ### 🌐 Dataset Source

    This dataset, **Online Shoppers' Intention**, is publicly available on the **UCI Machine Learning Repository**. It contains valuable insights into user behavior, helping to predict whether a user will complete an online purchase.

    You can explore the dataset in more detail by visiting the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).  
    """)

    # Añadir espacio entre bibliografía y herramientas
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # Herramientas utilizadas
    st.markdown("""
    ### 🛠️ Tools Used
    """)
    
    cols = st.columns(4)
    with cols[0]:
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", width=45)
    with cols[1]:
        st.image("https://code.visualstudio.com/assets/favicon.ico", width=45)
    with cols[2]:
        st.image("https://jupyter.org/assets/homepage/main-logo.svg", width=45)
    with cols[3]:
        # Ajustar el tamaño del logo de Streamlit
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=102)



elif opcion == "EDA Univariate":
    st.title("📊 EDA Univariate")

    # Añadir el texto explicativo antes de los gráficos
    st.markdown("""
    We will now explore some univariate charts to gain insights into individual features of the dataset. These visualizations will help us understand key trends, distributions, and patterns, providing a clearer view of how different aspects of user behavior contribute to our predictions. You will find below the most relevant charts that will help in the analysis.
    """)

    # Ruta del archivo CSV
    file_path = r"C:\\Users\\julyj\\OneDrive\\Desktop\\IRONHACK\\Proyecto Final\\online_shoppers_intentions_clean.csv"

    try:
        data = pd.read_csv(file_path)

        # Función para mostrar u ocultar gráficos
        def toggle_chart(chart_name, chart_func):
            if chart_name not in st.session_state:
                st.session_state[chart_name] = False
            if st.button(f"{chart_name}"):
                st.session_state[chart_name] = not st.session_state[chart_name]
            if st.session_state[chart_name]:
                chart_func()

        # Definición de gráficos
        def region_chart():
            # Crear gráfico de barras para la región
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=data['Region'], palette=['#FF9999'], ax=ax)
            ax.set_title("Frequency of Region", fontsize=14, fontweight='bold')

            # Mapeo de los números de región a nombres
            region_mapping = {
                1: "North America",
                2: "Europe",
                3: "Asia",
                4: "South America",
                5: "Africa",
                6: "Oceania",
                7: "Middle East",
                8: "Antarctica",
                9: "Other"
            }

            # Crear un DataFrame para la tabla con la asignación de valores
            region_data = pd.DataFrame(list(region_mapping.items()), columns=["Region", "Region Name"])
            region_data = region_data.set_index("Region")
            
            # Mostrar el gráfico y la tabla en columnas
            col1, col2 = st.columns([3, 1])  # 3 para el gráfico, 1 para la tabla
            with col1:
                st.pyplot(fig)
            with col2:
                st.markdown("### Regions")
                st.table(region_data.style.background_gradient(cmap="Greys"))

        def revenue_chart():
            fig, ax = plt.subplots()
            sns.countplot(x=data['Revenue'], palette=['#FF9999'], ax=ax)
            ax.set_title("Frequency of Revenue", fontsize=14, fontweight='bold')
            st.pyplot(fig)

        def month_chart():
            fig, ax = plt.subplots()
            sns.countplot(x=data['Month'], palette=['#FF9999'], order=data['Month'].value_counts().index, ax=ax)
            ax.set_title("Frequency of Month", fontsize=14, fontweight='bold')
            st.pyplot(fig)

        def weekend_chart():
            fig, ax = plt.subplots()
            sns.countplot(x=data['Weekend'], palette=['#FF9999'], ax=ax)
            ax.set_title("Frequency of Weekend", fontsize=14, fontweight='bold')
            st.pyplot(fig)

        # Mostrar gráficos
        toggle_chart("📊 Bar Chart: Revenue", revenue_chart)
        toggle_chart("📊 Bar Chart: Month", month_chart)
        toggle_chart("📊 Bar Chart: Region", region_chart)
        toggle_chart("📊 Bar Chart: Weekend", weekend_chart)

    except FileNotFoundError:
        st.error(f"The file was not found at the specified path: {file_path}")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
elif opcion == "EDA Bivariate":
    st.title("📊 EDA Bivariate")

    # Añadir el texto explicativo antes de los gráficos
    st.markdown("""
    Now, let's take a closer look at the relationships between two variables at a time. Bivariate analysis allows us to uncover patterns, correlations, and interactions between features. By examining these relationships, we can better understand how different factors influence each other, and how they impact the likelihood of a customer’s behavior. You will find below the most relevant bivariate charts for a deeper analysis.
    """)

    # Ruta del archivo CSV
    file_path = r"C:\\Users\\julyj\\OneDrive\\Desktop\\IRONHACK\\Proyecto Final\\online_shoppers_intentions_clean.csv"

    try:
        data = pd.read_csv(file_path)

        # Función para mostrar u ocultar gráficos
        def toggle_chart(chart_name, chart_func):
            # Reemplazar snake_case por espacio en el nombre del gráfico
            chart_name_display = chart_name.replace('_', ' ')  # Cambiar el nombre con espacios
            if chart_name not in st.session_state:
                st.session_state[chart_name] = False
            if st.button(f"{chart_name_display}"):  # Mostrar el nombre con espacios
                st.session_state[chart_name] = not st.session_state[chart_name]
            if st.session_state[chart_name]:
                chart_func()

        # Definición de gráficos
        def revenue_month_chart():
            fig, ax = plt.subplots()
            pd.crosstab(data['Month'], data['Revenue']).plot(kind='bar', stacked=True, color=['#99ff99', '#66cc66'], ax=ax)
            ax.set_title("Revenue by Month (Stacked)", fontsize=14)
            st.pyplot(fig)

        def visitor_revenue_chart():
            fig, ax = plt.subplots()
            pd.crosstab(data['Visitor_type'], data['Revenue']).plot(kind='bar', color=['#99ff99', '#66cc66'], ax=ax)
            ax.set_title("Visitor Type by Revenue", fontsize=14)
            st.pyplot(fig)

        def bounce_rate_violin_chart():
            fig, ax = plt.subplots()
            sns.violinplot(data=data, x='Revenue', y='Bounce_rates', palette='Greens', ax=ax)
            ax.set_title("Bounce Rates by Revenue", fontsize=14)
            st.pyplot(fig)

        # Mostrar gráficos
        toggle_chart("📊 Stacked Bar Chart: Revenue with Month", revenue_month_chart)
        toggle_chart("📊 Bar Chart: Visitor Type with Revenue", visitor_revenue_chart)
        toggle_chart("📊 Violin Plot: Revenue with Bounce Rates", bounce_rate_violin_chart)

    except FileNotFoundError:
        st.error(f"The file was not found at the specified path: {file_path}")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")


elif opcion == "Machine Learning":
    # Definición de modelos y sus imágenes
    modelos = {
        "Owl-AI (Logistic Regression)": LogisticRegression,
        "Panther-Trees (Random Forest)": RandomForestClassifier,
        "Cheetah-Boost (XGBoost)": XGBClassifier,
        "Fox-Gradient (Gradient Boosting)": GradientBoostingClassifier,
        "Eagle-Learner (AdaBoost)": AdaBoostClassifier,
        "🎅 Eagle-Learner Merry Christmas (AdaBoost + GridSearch)": AdaBoostClassifier
    }

    imagenes = {
        "Owl-AI (Logistic Regression)": r"C:\\Users\\julyj\\Downloads\\DALL·E 2024-12-11 20.44.44 - A highly adorable and futuristic owl with large glowing eyes, a shiny metallic body, and soft, rounded features. The owl should have a cute and friend.webp",
        "Panther-Trees (Random Forest)": r"C:\\Users\\julyj\\Downloads\\DALL·E 2024-12-11 20.44.51 - A futuristic panther with sleek metallic features, glowing elements, and soft, rounded edges. The panther should look powerful yet approachable, with .webp",
        "Cheetah-Boost (XGBoost)": r"C:\\Users\\julyj\\Downloads\\DALL·E 2024-12-11 20.44.56 - An adorable futuristic cheetah with glowing stripes, a sleek metallic body, and a playful expression. The cheetah should look agile and fast, with dyn.webp",
        "Fox-Gradient (Gradient Boosting)": r"C:\\Users\\julyj\\Downloads\\DALL·E 2024-12-11 20.45.00 - An adorable futuristic fox with a shiny metallic coat, glowing gradient patterns, and a curious expression. The fox should appear agile and sleek, wit.webp",
        "Eagle-Learner (AdaBoost)": r"C:\\Users\\julyj\\Downloads\\DALL·E 2024-12-11 20.45.05 - An adorable futuristic eagle with glowing wings, metallic feathers, and a majestic but friendly expression. The eagle should appear soaring in a brigh.webp",
        "🎅 Eagle-Learner Merry Christmas (AdaBoost + GridSearch)": r"C:\\Users\\julyj\\Downloads\\DALL·E 2024-12-11 20.45.12 - An adorable futuristic eagle with glowing wings and metallic feathers, similar in style to the previous Eagle-Learner design, wearing a Santa hat. The.webp"
    }

    st.markdown("### 🌌 Elige a tu héroe cibernético para el análisis.")

    # Lógica para desbloquear el modelo "Eagle-Learner Merry Christmas"
    if "entrenado_ada" not in st.session_state:
        st.session_state["entrenado_ada"] = False
    if "desbloqueado" not in st.session_state:
        st.session_state["desbloqueado"] = False
    if "mostrar_mensaje" not in st.session_state:
        st.session_state["mostrar_mensaje"] = False

    modelos_disponibles = list(modelos.keys())[:-1]
    if st.session_state["desbloqueado"]:
        modelos_disponibles.append("🎅 Eagle-Learner Merry Christmas (AdaBoost + GridSearch)")

    modelo_seleccionado = st.selectbox("Modelos disponibles:", modelos_disponibles)

    # Mostrar imagen del modelo
    st.image(imagenes[modelo_seleccionado], use_column_width=True)
    st.markdown(f"### Modelo Seleccionado: {modelo_seleccionado}")

    # Entrenamiento del modelo
    if st.button("Entrenar Modelo"):
        Modelo = modelos[modelo_seleccionado]()
        file_path = r"C:\\Users\\julyj\\OneDrive\\Desktop\\IRONHACK\\Proyecto Final\\online_shoppers_intention.csv"
        data = pd.read_csv(file_path)

        # Preprocesamiento de datos
        label_encoders = {}
        for col in ['Month', 'VisitorType']:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le

        X = data.drop('Revenue', axis=1)
        y = data['Revenue']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        Modelo.fit(X_train, y_train)
        y_pred = Modelo.predict(X_test)

        st.markdown(f"### Precisión del Modelo: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        if modelo_seleccionado == "Eagle-Learner (AdaBoost)":
            st.session_state["entrenado_ada"] = True

        if modelo_seleccionado == "🎅 Eagle-Learner Merry Christmas (AdaBoost + GridSearch)":
            st.session_state["mostrar_mensaje"] = True

    # Desbloqueo al final
    if st.session_state["entrenado_ada"] and not st.session_state["desbloqueado"]:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("🎁 ¡Un regalo especial para ti está listo!")
        if st.button("Desbloquear Eagle-Learner Merry Christmas"):
            st.session_state["desbloqueado"] = True
            st.balloons()
            st.success("¡Felicidades! Has desbloqueado a 🎅 Eagle-Learner Merry Christmas, nuestro modelo más avanzado y festivo.")

    # Mostrar el mensaje especial con plumas y botón
    if st.session_state["mostrar_mensaje"]:
        st.markdown("""<div class='feather'>🪶</div>""", unsafe_allow_html=True)
        st.markdown("🎅 Eagle-Learner Merry Christmas tiene algo que decirte. Pulsa el botón para descubrir el mensaje.")
        if st.button("Apriétame", key="mensaje_boton"):
            st.markdown("""
            <div class="message-card">
            "Esta Navidad, recuerda que las personas que conoces en los momentos más inesperados pueden cambiar tu vida para siempre.<br>
            Cada encuentro, como cada dato en tus análisis, tiene un propósito especial.<br>
            Gracias por dejarme acompañarte en esta aventura.<br>
            <strong>¡Feliz Navidad a todos, y que vuestro futuro esté lleno de magia y predicciones acertadas!</strong>"
            </div>
            """, unsafe_allow_html=True)



elif opcion == "Greetings":
    st.title("🎉 Greetings")

    # Main message (waiting)
    st.markdown("**Wait.. Is there another gift???**")

    # Add a larger emoji button
    st.markdown(
        """
        <style>
            .gift-button {
                font-size: 200px;  /* Increase the emoji size */
                background: none;
                border: none;
                cursor: pointer;
                display: inline-block;
                text-align: center;
                padding: 0;
            }
        </style>
        """, unsafe_allow_html=True)

    # Button to trigger the gif
    if st.button("🎁", help="Press for a special gift!"):
        # Show the GIF when the button is pressed
        st.image("C:\\Users\\julyj\Downloads\\4ddecf21686dcdc398a17448c2d91aaa.jpg", use_column_width=True)  # Updated path for the GIF