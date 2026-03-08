import streamlit as st
import joblib
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer # Importación más específica
from nltk.tokenize import RegexpTokenizer

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Clasificador ODS AI", page_icon="🌍")

# Descargas necesarias de NLTK
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk()

# --- COMPONENTES DEL PREPROCESAMIENTO ---
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')
tokenizer = RegexpTokenizer(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]+')

# IMPORTANTE: El nombre de esta función DEBE ser idéntico al del Notebook
def preprocesar_texto(texto):
    if not isinstance(texto, str):
        return ""
    tokens = tokenizer.tokenize(texto.lower())
    tokens_limpios = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(tokens_limpios)

# --- DICCIONARIO DE MAPEO ODS ---
diccionario_ods = {
    1: "Fin de la pobreza", 2: "Hambre cero", 3: "Salud y bienestar",
    4: "Educación de calidad", 5: "Igualdad de género", 6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante", 8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura", 10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles", 12: "Producción y consumo responsables",
    13: "Acción por el clima", 14: "Vida submarina", 15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas"
}

# --- CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    # Verificamos si el archivo existe antes de intentar abrirlo
    if not os.path.exists('modelo_ods_final.pkl'):
        st.error("Archivo 'modelo_ods_final.pkl' no encontrado.")
        st.stop()
    
    with open('modelo_ods_final.pkl', 'rb') as f:
        return joblib.load(f)

model = cargar_modelo()

# --- INTERFAZ DE USUARIO ---
st.title("🌍 Clasificador IA de Objetivos de Desarrollo Sostenible (ODS)")

texto_usuario = st.text_area("Ingresa el texto a analizar:", height=200, placeholder="Ejemplo: Proyecto de reforestación...")

if st.button("Clasificar Texto"):
    if texto_usuario.strip():
        try:
            # Predicción
            prediccion_num = model.predict([texto_usuario])[0]
            probs = model.predict_proba([texto_usuario])[0]
            
            nombre_ods_principal = diccionario_ods.get(prediccion_num, "ODS no identificado")
            confianza_principal = max(probs)

            # Mostrar Resultado Principal
            with st.container():
                st.success(f"### ✅ Clasificación Exitosa") # Título del bloque
                
                col_info, col_metric = st.columns([2, 1])
                
                with col_info:
                    st.markdown(f"#### **ODS {prediccion_num}: {nombre_ods_principal}**")
                    st.write("El texto analizado coincide con los lineamientos de este objetivo.")
                    
                with col_metric:
                    st.metric(label="Confianza", value=f"{confianza_principal:.2%}")

            # --- 4. ANÁLISIS DE RELEVANCIA (TOP 3 CON COLORES) ---
            st.markdown("---")
            st.write("#### 📊 Desglose de Relevancia")
            st.caption("Probabilidad asignada por el modelo a los objetivos más relacionados:")

            top_3_indices = probs.argsort()[-3:][::-1]

            for idx in top_3_indices:
                ods_n = model.classes_[idx]
                prob_n = probs[idx]
                nombre_n = diccionario_ods.get(ods_n, "Desconocido")
                
                # Mostramos el nombre y el porcentaje en una sola línea
                st.write(f"**ODS {ods_n}**: {nombre_n} ({prob_n:.1%})")
                
                # UX: Cambiamos el color de la barra según la probabilidad
                # Si es el principal (el primero del top 3), usamos el color por defecto (azul)
                # Si es secundario, la barra es más tenue
                st.progress(float(prob_n))
        except Exception as e:
            st.error(f"Error durante la clasificación: {e}")
    else:
        st.warning("Por favor, ingresa un texto.")