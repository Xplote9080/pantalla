import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os
import xml.etree.ElementTree as ET

from script2 import (
    cargar_estaciones,
    interpolar_puntos,
    obtener_elevaciones_paralelo,
    calcular_pendiente_suavizada
)

st.set_page_config(page_title="Pantalla Cabina", layout="wide")
st.title("🚆 Vista estilo cabina - Perfil altimétrico en tiempo real")

st.markdown("""
Cargá un archivo **CSV** o **KML** con estaciones (Nombre, Km, Lat, Lon). El sistema mostrará el perfil altimétrico alrededor del kilómetro actual.

También podés elegir un subtramo del archivo y cuántos *workers* usar para consultar elevaciones. Más workers = más rápido, pero puede ser rechazado por la API (Error 429).
""")

# --- Procesar KML ---
def procesar_kml(kml_file):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_file)
    root = tree.getroot()
    placemarks = root.findall('.//kml:Placemark', ns)
    datos = []
    for pm in placemarks:
        name_tag = pm.find('kml:name', ns)
        point_tag = pm.find('.//kml:Point', ns)
        coord_tag = point_tag.find('kml:coordinates', ns) if point_tag is not None else None
        if name_tag is not None and coord_tag is not None:
            try:
                texto = name_tag.text.strip()
                nombre, km = texto.split(',')
                lon, lat, *_ = coord_tag.text.strip().split(',')
                datos.append({
                    'Nombre': nombre.strip(),
                    'Km': float(km.strip()),
                    'Lat': float(lat),
                    'Lon': float(lon)
                })
            except:
                continue
    return pd.DataFrame(datos)

# --- Cargar archivo ---
archivo_subido = st.file_uploader("📤 Subí tu archivo CSV o KML", type=["csv", "kml"])
df_estaciones = pd.DataFrame()

if archivo_subido:
    if archivo_subido.name.endswith(".csv"):
        df_estaciones = pd.read_csv(archivo_subido)
    elif archivo_subido.name.endswith(".kml"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
            tmp.write(archivo_subido.read())
            df_estaciones = procesar_kml(tmp.name)
            os.remove(tmp.name)

# --- Validación y selección de subtramo ---
if not df_estaciones.empty:
    df_estaciones.columns = [c.strip().capitalize() for c in df_estaciones.columns]
    if set(['Nombre', 'Km', 'Lat', 'Lon']).issubset(df_estaciones.columns):
        st.success("✅ Datos válidos cargados")

        st.subheader("📍 Selección del tramo a visualizar")

        estaciones_ordenadas = df_estaciones.sort_values("Km").reset_index(drop=True)
        nombres_estaciones = estaciones_ordenadas["Nombre"].tolist()

        est_inicio = st.selectbox("Estación inicial", nombres_estaciones, index=0)
        est_fin = st.selectbox("Estación final", nombres_estaciones, index=len(nombres_estaciones)-1)

        km_inicio = estaciones_ordenadas.loc[estaciones_ordenadas["Nombre"] == est_inicio, "Km"].values[0]
        km_fin = estaciones_ordenadas.loc[estaciones_ordenadas["Nombre"] == est_fin, "Km"].values[0]

        if km_inicio >= km_fin:
            st.error("❌ La estación inicial debe estar antes que la final (según el kilómetro)")
            st.stop()

        df_tramo = estaciones_ordenadas[(estaciones_ordenadas["Km"] >= km_inicio) & (estaciones_ordenadas["Km"] <= km_fin)].copy()
        st.success(f"✅ {len(df_tramo)} estaciones seleccionadas entre {est_inicio} y {est_fin}")

        st.subheader("⚙️ Parámetros de visualización")
        intervalo = st.slider("Intervalo de interpolación (m)", 50, 500, 100, step=10)
        ventana_km = st.slider("Rango visible (km)", 1, 10, 5)
        km_actual = st.slider("Kilómetro actual", float(km_inicio), float(km_fin), (km_inicio + km_fin)/2, step=0.1)
        max_workers = st.slider("👷‍♂️ Número de workers para consultas a la API", 1, 10, 2,
                                help="Cuantos más workers, más rápido será, pero aumenta el riesgo de que la API rechace las consultas (Error 429: Too Many Requests).")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            df_tramo.to_csv(tmp_csv.name, index=False)
            estaciones = cargar_estaciones(tmp_csv.name)

        puntos_interp = interpolar_puntos(estaciones, intervalo_m=intervalo)
        puntos_interp = obtener_elevaciones_paralelo(
            puntos_interp,
            author="LAL",
            progress_callback=st.progress(0).progress,
            max_workers=max_workers
        )

        kms = np.array([p.km for p in puntos_interp])
        elevs = np.array([p.elevation for p in puntos_interp])
        mask = (kms >= km_actual - ventana_km) & (kms <= km_actual + ventana_km)
        kms_vis = kms[mask]
        elevs_vis = elevs[mask]
        pendientes = calcular_pendiente_suavizada(kms_vis, elevs_vis, window_length=5)

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=kms_vis, y=elevs_vis, mode='lines', line=dict(color='cyan', width=3), name='Perfil'))
        fig.add_shape(type='line', x0=km_actual, x1=km_actual, y0=min(elevs_vis), y1=max(elevs_vis), line=dict(color='red', width=3, dash='dot'))

        for i in range(1, len(kms_vis)):
            mid_km = (kms_vis[i] + kms_vis[i-1]) / 2
            mid_elev = (elevs_vis[i] + elevs_vis[i-1]) / 2
            pendiente = pendientes[i]
            if not np.isnan(pendiente):
                fig.add_annotation(x=mid_km, y=mid_elev+1, text=f"{pendiente:+.1f}", showarrow=False, font=dict(size=14, color='white'))

        tramo = f"{est_inicio} - {est_fin}"
        fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white', size=18),
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(title='Kilómetro', color='white', showgrid=False),
            yaxis=dict(title='Elevación (m)', color='white', showgrid=False),
            title=dict(text=f"{tramo} | Km actual: {km_actual:.3f}", x=0.5, xanchor='center')
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("❌ El archivo debe tener las columnas: Nombre, Km, Lat, Lon")
