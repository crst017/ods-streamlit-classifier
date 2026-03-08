import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Clasificador ODS AI", page_icon="🌍")

# Descargas necesarias de NLTK (Streamlit Cloud las ejecutará al iniciar)
nltk.download('stopwords')

# --- COMPONENTES DEL PREPROCESAMIENTO ---
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')
tokenizer = RegexpTokenizer(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]+')

def preprocesar_texto_final(texto):
    if not isinstance(texto, str):
        return ""
    tokens = tokenizer.tokenize(texto.lower())
    # Filtrar stopwords y aplicar stemming en un solo paso
    tokens_limpios = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(tokens_limpios)

# --- DICCIONARIO DE MAPEO ODS ---
diccionario_ods = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas"
}

# --- CARGA DEL MODELO ---
@st.cache_resource # Esto evita que el modelo se recargue cada vez que el usuario hace clic
def cargar_modelo():
    with open('modelo_ods_final.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# --- INTERFAZ DE USUARIO ---
st.title("🌍 Clasificador de Objetivos de Desarrollo Sostenible (ODS)")
st.markdown("""
Esta herramienta utiliza Inteligencia Artificial para identificar a qué ODS pertenece un texto. 
Pega un párrafo descriptivo de un proyecto o iniciativa abajo.
""")

texto_usuario = st.text_area(
    "Ingresa el texto a analizar:",
    placeholder="Ejemplo: Instalación de sistemas de riego eficiente para agricultores locales...",
    height=200
)

if st.button("Clasificar Texto"):
    if texto_usuario.