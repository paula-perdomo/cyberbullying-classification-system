import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate

# ==============================
# CONFIGURACIÓN
# ==============================
# Cargar variables de entorno desde .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Modelo
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

# Esquema de salida
response_schemas = [
    ResponseSchema(
        name="es_cyberbullying",
        description="Indica si el texto contiene cyberbullying. Valores: 'Sí' o 'No'."
    ),
    ResponseSchema(
        name="categoria",
        description="Categoría: género, edad, religión, raza, u otro tipo. Si no aplica, 'N/A'."
    ),
    ResponseSchema(
        name="justificacion",
        description="Explicación breve y profesional."
    )
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en detección de ciberacoso."),
    ("user", 
     "Texto a analizar: {texto}\n\n"
     "Responde en el formato:\n{format_instructions}")
])

# Función de análisis
def analizar_texto(texto: str):
    _prompt = prompt.format_messages(texto=texto, format_instructions=format_instructions)
    response = llm(_prompt)
    return parser.parse(response.content)

# ==============================
# INTERFAZ STREAMLIT
# ==============================
st.set_page_config(page_title="Detección de Cyberbullying", page_icon=None, layout="centered")

# Logo en la esquina superior izquierda con columnas
col1, col2 = st.columns([1, 8])
with col1:
    st.image("maia_uniandes.png", width=80)  # Logo local
with col2:
    st.title("Detección de Cyberbullying")

st.sidebar.title("Estadísticas de Cyberbullying")
st.sidebar.image("cyberbullying_graph.png", use_container_width=True)
st.sidebar.markdown(""" 
El gráfico muestra el porcentaje de personas víctimas de cyberacoso por red social. Es importante que tu aplicación detecte a tiempo contenido violento. Entrenamos modelos con datos reales y una alta sensibilidad, que ahora están a tu servicio.
""")

st.markdown("""
La proliferación de la comunicación digital ha provocado un aumento significativo del ciberacoso, lo que representa una grave amenaza para el bienestar de los usuarios en línea.  
El **73% de usuarios de X (antes Twitter)** reportó haber sido víctima de ciberacoso, haciendo de esta plataforma la más hostil.  

Este proyecto, el **"Sistema de Clasificación de Ciberacoso" (Cyberbullying Classification System)**, aborda este desafío mediante el desarrollo de un modelo robusto de aprendizaje automático supervisado para detectar y categorizar automáticamente diversas formas de ciberacoso a partir de datos de texto.  

Te invitamos a probar nuestra API y corroborar si tu mensaje se clasifica como ciberacoso y de qué tipo.
""")

st.write("Escribe un mensaje y detecta si contiene ciberacoso o no.")

# Inicializar sesión
if "resultado" not in st.session_state:
    st.session_state.resultado = None

# Entrada de texto
texto = st.text_area("Escribe el mensaje aquí:", height=150)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Analizar"):
        if texto.strip():
            st.session_state.resultado = analizar_texto(texto)
        else:
            st.warning("Por favor, ingresa un texto.")

with col2:
    if st.button("Reiniciar"):
        st.session_state.resultado = None
        st.experimental_rerun()

# Mostrar resultado
if st.session_state.resultado:
    resultado = st.session_state.resultado
    st.subheader("Resultado del análisis")
    st.success(f"**¿Es cyberbullying?** {resultado['es_cyberbullying']}")
    st.info(f"**Categoría:** {resultado['categoria']}")
    st.write(f"**Justificación:** {resultado['justificacion']}")
