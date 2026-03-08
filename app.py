import streamlit as st
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import nltk

# Descargas necesarias para el servidor de Streamlit
nltk.download('stopwords')

# --- COPIA AQUÍ TU FUNCIÓN DE PREPROCESAMIENTO DEL HTML ---
stop_words_es = set(stopwords.words('spanish'))
regex_tokenizer = RegexpTokenizer(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]+')
stemmer = SnowballStemmer('spanish')

def preprocesar_texto(texto):
    texto = str(texto).lower()
    tokens = regex_tokenizer.tokenize(texto)
    tokens_procesados = [stemmer.stem(word) for word in tokens if word not in stop_words_es]
    return ' '.join(tokens_procesados)

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Clasificador ODS", page_icon="🌍")
st.title("Clasificador de texto según Objetivos de Desarrollo Sostenible (ODS)")

@st.cache_resource
def load_model():
    return joblib.load('modelo_ods_final.pkl')

model = load_model()

texto_usuario = st.text_area("Ingrese el texto a clasificar:", height=150)

if st.button("Clasificar Texto"):
    if texto_usuario:
        prediccion = model.predict([texto_usuario])[0]
        probs = model.predict_proba([texto_usuario])[0]
        confianza = max(probs)
        
        st.subheader(f"Resultado: Tópico {prediccion}")
        st.write(f"**Confianza del modelo:** {confianza:.2%}")
        
        # Mostrar gráfico de barras de confianza
        df_probs = pd.DataFrame({
            'Tópico': model.classes_,
            'Probabilidad': probs
        })
        st.bar_chart(df_probs.set_index('Tópico'))
    else:
        st.error("Por favor, ingrese un texto.")