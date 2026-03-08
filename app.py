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

def preprocesar_texto(texto):
    if not isinstance(texto, str):
        return ""
    tokens = tokenizer.tokenize(texto.lower())
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
    if texto_usuario.strip():
        # 1. Predicción de clase y probabilidades
        prediccion_num = model.predict([texto_usuario])[0]
        probs = model.predict_proba([texto_usuario])[0]
        
        # 2. Obtener nombre del ODS principal
        nombre_ods_principal = diccionario_ods.get(prediccion_num, "ODS no identificado")
        confianza_principal = max(probs)

        # 3. Mostrar Resultado Principal
        st.success(f"### Resultado Principal: ODS {prediccion_num}")
        st.subheader(f"**{nombre_ods_principal}**")
        st.metric("Nivel de Confianza", f"{confianza_principal:.2%}")

        # 4. Mostrar Top 3 de ODS relacionados
        st.divider()
        st.write("#### Análisis de relevancia (Top 3):")
        
        # Obtener los índices de los 3 valores más altos
        top_3_indices = probs.argsort()[-3:][::-1]
        
        for idx in top_3_indices: 
            ods_n = model.classes_[idx]
            prob_n = probs[idx]
            nombre_n = diccionario_ods.get(ods_n, "Desconocido")
            
            # Crear columnas para una visualización limpia
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**ODS {ods_n}**: {nombre_n}")
                st.progress(float(prob_n))
            with col2:
                st.write(f"{prob_n:.1%}")
                
        if confianza_principal < 0.35:
            st.warning("⚠️ **Nota:** La confianza es baja. El texto podría ser muy corto o tocar múltiples temas a la vez.")
            
    else:
        st.warning("Por favor, ingresa un texto para poder clasificarlo.")

# --- PIE DE PÁGINA ---
st.sidebar.info("Modelo entrenado con el dataset OSDG Community (Zenodo).")