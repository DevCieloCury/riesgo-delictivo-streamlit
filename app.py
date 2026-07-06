import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date
from xgboost import XGBClassifier

# ============================================================
# Configuración general de la aplicación
# ============================================================
st.set_page_config(
    page_title="Predicción de Riesgo Delictivo",
    page_icon="📊",
    layout="centered"
)

# ============================================================
# Estilos visuales simples
# ============================================================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 18px;
    color: #CBD5E1;
    margin-bottom: 25px;
}
.info-box {
    background-color: #12304A;
    padding: 18px;
    border-radius: 10px;
    color: #DCEBFF;
    margin-bottom: 25px;
}
.result-card {
    padding: 22px;
    border-radius: 12px;
    margin-top: 20px;
    margin-bottom: 20px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}
.risk-low {
    background-color: #143D2B;
    color: #77DD99;
}
.risk-medium {
    background-color: #4A3A12;
    color: #FFD166;
}
.risk-high {
    background-color: #4A1515;
    color: #FF6B6B;
}
.small-note {
    color: #94A3B8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Carga automática del paquete y modelo
# ============================================================
@st.cache_resource
def cargar_recursos():
    ruta_paquete = Path("paquete_streamlit.pkl")
    ruta_modelo = Path("modelo_xgboost.json")

    if not ruta_paquete.exists():
        raise FileNotFoundError("No se encontró paquete_streamlit.pkl en la carpeta de la aplicación.")

    if not ruta_modelo.exists():
        raise FileNotFoundError("No se encontró modelo_xgboost.json en la carpeta de la aplicación.")

    paquete = joblib.load(ruta_paquete)

    modelo = XGBClassifier()
    modelo.load_model(str(ruta_modelo))

    return paquete, modelo

# ============================================================
# Encabezado principal
# ============================================================
st.markdown(
    '<div class="main-title">Prototipo de predicción de nivel de riesgo delictivo</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Aplicación simple para estimar el NIVEL_RIESGO a partir de una combinación espacio-temporal.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
<b>Unidad de predicción:</b> esta aplicación no predice el riesgo de una persona individual.
El modelo estima el nivel de riesgo para una combinación de <b>área policial, fecha y franja horaria</b>.
</div>
""", unsafe_allow_html=True)

# ============================================================
# Intento de carga del modelo
# ============================================================
try:
    paquete, modelo = cargar_recursos()

    preprocessor = paquete["preprocessor"]
    selector = paquete["selector"]
    modelo_nombre = paquete["modelo_nombre"]
    mapa_inverso = paquete["mapa_inverso"]
    orden_riesgo = paquete["orden_riesgo"]
    areas_trabajo = paquete["areas_trabajo"]

    st.sidebar.header("Estado del despliegue")
    st.sidebar.success("Modelo cargado correctamente")
    st.sidebar.write("Modelo:", modelo_nombre)
    st.sidebar.caption("Archivos cargados automáticamente desde el repositorio:")
    st.sidebar.caption("paquete_streamlit.pkl")
    st.sidebar.caption("modelo_xgboost.json")

    st.success(f"Modelo cargado correctamente: {modelo_nombre}")

except Exception as e:
    st.error("No se pudo cargar el modelo o el paquete de preprocesamiento.")
    st.write("Verifica que los archivos `paquete_streamlit.pkl` y `modelo_xgboost.json` estén en la misma carpeta que `app.py`.")
    st.exception(e)
    st.stop()

# ============================================================
# Formulario de entrada
# ============================================================
st.header("Datos de entrada")

st.write(
    "Selecciona el área policial, la fecha de consulta y la franja horaria. "
    "A partir de la fecha, la aplicación calcula automáticamente el mes, el día del mes, "
    "el día de semana y si corresponde a fin de semana."
)

col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("Área policial", areas_trabajo)

with col2:
    franja = st.selectbox(
        "Franja horaria",
        ["Madrugada", "Mañana", "Tarde", "Noche"]
    )

fecha_consulta = st.date_input(
    "Fecha de consulta",
    value=date(2023, 6, 15)
)

# Variables derivadas desde la fecha seleccionada
mes = fecha_consulta.month
dia = fecha_consulta.day
dia_semana = fecha_consulta.weekday()  # 0=Lunes, 6=Domingo
es_fin_semana = 1 if dia_semana in [5, 6] else 0

st.subheader("Variables calculadas desde la fecha")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Mes", mes)
col_b.metric("Día", dia)
col_c.metric("Día semana", dia_semana)
col_d.metric("Fin de semana", "Sí" if es_fin_semana == 1 else "No")

# ============================================================
# Predicción
# ============================================================
if st.button("Predecir nivel de riesgo"):

    nuevo_registro = pd.DataFrame([{
        "AREA NAME": area,
        "TIME RANGE": franja,
        "MONTH OCC": mes,
        "DAY OCC": dia,
        "DAYOFWEEK OCC": dia_semana,
        "IS_WEEKEND": es_fin_semana
    }])

    nuevo_pre = preprocessor.transform(nuevo_registro)
    nuevo_selected = selector.transform(nuevo_pre)

    prediccion_num = int(modelo.predict(nuevo_selected)[0])
    probabilidades = modelo.predict_proba(nuevo_selected)[0]

    riesgo_predicho = mapa_inverso[prediccion_num]

    if riesgo_predicho == "BAJO":
        clase_css = "risk-low"
    elif riesgo_predicho == "MEDIO":
        clase_css = "risk-medium"
    else:
        clase_css = "risk-high"

    st.markdown(
        f"""
        <div class="result-card {clase_css}">
            Nivel de riesgo predicho: {riesgo_predicho}
        </div>
        """,
        unsafe_allow_html=True
    )

    tabla_probabilidades = pd.DataFrame({
        "Nivel": orden_riesgo,
        "Probabilidad": probabilidades
    })

    tabla_probabilidades["Probabilidad"] = tabla_probabilidades["Probabilidad"].round(4)

    st.subheader("Probabilidades por clase")
    st.dataframe(tabla_probabilidades, use_container_width=True)
    st.bar_chart(tabla_probabilidades.set_index("Nivel"))

    st.subheader("Interpretación")

    if riesgo_predicho == "BAJO":
        st.write(
            "La combinación ingresada se parece más a los casos históricos clasificados como riesgo BAJO. "
            "Esto no significa ausencia total de delitos, sino menor concentración relativa según el modelo."
        )
    elif riesgo_predicho == "MEDIO":
        st.write(
            "La combinación ingresada se ubica en un nivel intermedio. "
            "Esto indica que el modelo no la asocia claramente con los extremos BAJO o ALTO."
        )
    else:
        st.write(
            "La combinación ingresada presenta mayor similitud con los casos históricos clasificados como riesgo ALTO. "
            "Este resultado debe interpretarse como una alerta del modelo para esa combinación espacio-temporal."
        )
