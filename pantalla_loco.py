import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os
import xml.etree.ElementTree as ET

# Importar solo las funciones necesarias
from script2 import (
    cargar_estaciones,
    interpolar_puntos,
    obtener_elevaciones_paralelo,
    calcular_pendiente_suavizada,
    load_cache_to_memory # Importar la función de carga de caché
)

st.set_page_config(page_title="Pantalla Cabina", layout="wide")
st.title("🚆 Vista estilo cabina - Perfil altimétrico en tiempo real")

st.markdown("""
Cargá un archivo **CSV** o **KML** con estaciones (Nombre, Km, Lat, Lon). El sistema mostrará el perfil altimétrico alrededor del kilómetro actual.

También podés elegir un subtramo del archivo y cuántos *workers* usar para consultar elevaciones. Más workers = más rápido, pero puede ser rechazado por la API (Error 429).
""")

# --- Inicializar Caché de Elevaciones (Usando Streamlit's caching) ---
# Usamos st.cache_resource para cargar el caché solo una vez por despliegue
# Esto llama a load_cache_to_memory() en script2.py la primera vez que se ejecuta
# y reutiliza el resultado en ejecuciones posteriores.
@st.cache_resource
def initialize_elevation_cache():
    """Inicializa y carga el caché de elevaciones usando la función del script2."""
    # La función load_cache_to_memory maneja la lógica de carga desde archivo
    return load_cache_to_memory()

# Llamar para inicializar el caché
elevation_cache = initialize_elevation_cache()
# Asegurarse de que script2 use esta instancia del caché
# (Aunque load_cache_to_memory está diseñada para usar la variable global _cache
# dentro de script2, asignarla explícitamente aquí puede añadir claridad
# o ser necesario dependiendo de cómo se manejen los módulos en Streamlit.
# Sin embargo, la implementación actual de load_cache_to_memory y _save_elevation_to_cache
# en script2.py que modifican la variable global _cache debería funcionar bien
# en el contexto de un solo proceso de Streamlit).
# Si tuvieras problemas de concurrencia o acceso al caché, podrías necesitar
# un mecanismo de bloqueo o pasar el objeto cache explícitamente.
# Por ahora, confiamos en la variable global dentro de script2.

# --- Procesar KML ---
# Esta función no necesita cambios
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
                # Manejar casos donde el texto no tiene ',' o tiene más de uno
                if ',' in texto:
                    nombre, km_str = texto.split(',', 1) # Split solo en la primera coma
                    nombre = nombre.strip()
                    km = float(km_str.strip())
                else:
                     nombre = texto.strip()
                     km = np.nan # O asignar un valor por defecto / saltar esta fila

                lon, lat, *alt = coord_tag.text.strip().split(',')
                lat = float(lat)
                lon = float(lon)
                # Opcionalmente, podrías intentar leer la altitud si existe en KML
                # altitude = float(alt[0]) if alt else np.nan

                if not np.isnan(km): # Solo añadir si se pudo parsear el KM
                    datos.append({
                        'Nombre': nombre,
                        'Km': km,
                        'Lat': lat,
                        'Lon': lon
                    })
            except ValueError as ve:
                 st.warning(f"Saltando entrada inválida en KML: '{name_tag.text}' - {ve}")
                 continue
            except Exception as e:
                 st.warning(f"Error inesperado al procesar Placemark: {e}")
                 continue
    return pd.DataFrame(datos)

# --- Cargar archivo ---
archivo_subido = st.file_uploader("📤 Subí tu archivo CSV o KML", type=["csv", "kml"])
df_estaciones = pd.DataFrame() # Inicializar vacío

if archivo_subido:
    # Usar io.BytesIO o io.StringIO para leer directamente de archivo_subido
    # sin guardar en disco, lo cual es más eficiente en Streamlit Cloud.
    # Para KML, como procesar_kml espera un archivo en disco, mantenemos tempfile por ahora.
    # Si procesar_kml pudiera leer de un objeto archivo, sería mejor.
    if archivo_subido.name.endswith(".csv"):
        try:
            # Leer directamente usando pandas
            df_estaciones = pd.read_csv(archivo_subido)
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            df_estaciones = pd.DataFrame() # Asegurar que esté vacío en caso de error

    elif archivo_subido.name.endswith(".kml"):
        # Para KML, todavía necesitamos un archivo en disco para ET.parse y simplekml
        # Idealmente, se reescribiría procesar_kml para trabajar con BytesIO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
            tmp.write(archivo_subido.getvalue()) # Usar getvalue() para obtener bytes
            tmp_path = tmp.name
        try:
            df_estaciones = procesar_kml(tmp_path)
        except Exception as e:
            st.error(f"Error al procesar el archivo KML: {e}")
            df_estaciones = pd.DataFrame() # Asegurar que esté vacío en caso de error
        finally:
            # Asegurarse de eliminar el archivo temporal
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# --- Validación y selección de subtramo ---
# Verificar si el DataFrame no está vacío y tiene las columnas requeridas
if not df_estaciones.empty and set(['Nombre', 'Km', 'Lat', 'Lon']).issubset(df_estaciones.columns):
    st.success(f"✅ Datos válidos cargados ({len(df_estaciones)} estaciones)")

    # Limpieza y ordenamiento del DataFrame cargado
    df_estaciones.columns = [c.strip().capitalize() for c in df_estaciones.columns]
    # Eliminar filas con valores faltantes en columnas críticas si es necesario
    df_estaciones.dropna(subset=['Km', 'Lat', 'Lon'], inplace=True)
    # Convertir a tipos numéricos, forzando errores a NaN y luego eliminando
    df_estaciones['Km'] = pd.to_numeric(df_estaciones['Km'], errors='coerce')
    df_estaciones['Lat'] = pd.to_numeric(df_estaciones['Lat'], errors='coerce')
    df_estaciones['Lon'] = pd.to_numeric(df_estaciones['Lon'], errors='coerce')
    df_estaciones.dropna(subset=['Km', 'Lat', 'Lon'], inplace=True)

    if df_estaciones.empty or len(df_estaciones) < 2:
        st.error("❌ No hay suficientes estaciones válidas con datos completos (Nombre, Km, Lat, Lon) después de la limpieza.")
        st.stop() # Detener la ejecución de esta parte

    # Ordenar estaciones por Km para la selección del tramo
    df_estaciones_ordenadas = df_estaciones.sort_values("Km").reset_index(drop=True)
    nombres_estaciones = df_estaciones_ordenadas["Nombre"].tolist()

    st.subheader("📍 Selección del tramo a visualizar")

    # Usar índices para los selectbox para evitar problemas si los nombres no son únicos
    idx_est_inicio = st.selectbox("Estación inicial", options=range(len(nombres_estaciones)), format_func=lambda x: nombres_estaciones[x], index=0)
    idx_est_fin = st.selectbox("Estación final", options=range(len(nombres_estaciones)), format_func=lambda x: nombres_estaciones[x], index=len(nombres_estaciones)-1)

    est_inicio = nombres_estaciones[idx_est_inicio]
    est_fin = nombres_estaciones[idx_est_fin]

    km_inicio = df_estaciones_ordenadas.loc[idx_est_inicio, "Km"]
    km_fin = df_estaciones_ordenadas.loc[idx_est_fin, "Km"]

    if km_inicio >= km_fin:
        st.error("❌ La estación inicial debe tener un kilómetro menor que la final.")
        st.stop()

    # Filtrar el DataFrame original (ordenado) por el tramo seleccionado
    df_tramo_seleccionado = df_estaciones_ordenadas[(df_estaciones_ordenadas["Km"] >= km_inicio) & (df_estaciones_ordenadas["Km"] <= km_fin)].copy()

    if df_tramo_seleccionado.empty:
         st.error("❌ El tramo seleccionado no contiene estaciones.")
         st.stop()

    st.success(f"✅ {len(df_tramo_seleccionado)} estaciones seleccionadas entre {est_inicio} (Km {km_inicio:.3f}) y {est_fin} (Km {km_fin:.3f})")

    st.subheader("⚙️ Parámetros de visualización y procesamiento")

    # Controles para parámetros
    intervalo = st.slider("Intervalo de interpolación (m)", 10, 500, 100, step=10,
                          help="Distancia en metros entre los puntos interpolados donde se consultará la elevación.")
    ventana_km = st.slider("Rango visible alrededor del Km actual (± Km)", 1, 10, 5,
                           help="Cuántos kilómetros antes y después del 'Kilómetro actual' se mostrarán en el gráfico.")
    # Rango del slider de Km actual basado en el tramo seleccionado
    km_actual = st.slider("Kilómetro actual", float(df_tramo_seleccionado['Km'].min()), float(df_tramo_seleccionado['Km'].max()),
                          float((df_tramo_seleccionado['Km'].min() + df_tramo_seleccionado['Km'].max())/2), step=0.1,
                          help="Simula la posición actual en la vía.")

    # Asegurarse de que km_actual esté dentro del rango del tramo visible
    km_actual = max(float(df_tramo_seleccionado['Km'].min()), min(float(df_tramo_seleccionado['Km'].max()), km_actual))


    max_workers = st.slider("👷‍♂️ Número de workers para consultas a la API", 1, 10, 2,
                            help="Cuantos más workers, más rápido será, pero aumenta el riesgo de que la API rechace las consultas (Error 429: Too Many Requests). Reducir este valor ayuda a evitar el error 429.")

    # Nuevo slider para la ventana de suavizado de la pendiente
    # Sugerir un rango y paso razonables. Debe ser un entero impar >= 3.
    window_length = st.slider("📏 Ventana de Suavizado para Pendiente (puntos)", 3, 51, 9, step=2,
                              help="Tamaño de la ventana para el filtro Savitzky-Golay al calcular la pendiente. Un valor más alto suaviza más la pendiente. Debe ser un número impar.")


    st.info("Procesando datos... Esto puede tardar unos minutos dependiendo del tramo y el intervalo.")

    # --- Procesamiento de Datos ---
    # Convertir el DataFrame del tramo a la lista de objetos Station
    # Ya no necesitamos el archivo temporal CSV aquí
    estaciones_tramo_list = [
        cargar_estaciones(df_tramo_seleccionado.copy()) # Pasar una copia para no modificar el DF original si la func lo hace
    ]
    # cargar_estaciones ahora devuelve una lista de Stations, no un DataFrame
    # La llamada cargar_estaciones(df_tramo_seleccionado) devolverá una lista con las estaciones del tramo

    # Solo necesitamos las estaciones del tramo para marcar en el gráfico, no para interpolar
    # La interpolación se hace sobre los puntos del tramo, no solo las estaciones
    # Usamos el df_tramo_seleccionado completo para la interpolación
    puntos_interp = interpolar_puntos(estaciones_tramo_list[0], intervalo_m=intervalo) # Pasar la lista de Stations

    # Usar un spinner para mostrar que se está procesando
    with st.spinner("Consultando elevaciones (puede tomar tiempo)..."):
        # Usar st.progress para la callback
        progress_bar = st.progress(0)
        puntos_con_elevacion = obtener_elevaciones_paralelo(
            puntos_interp,
            author="LAL", # Pasa tu autor
            progress_callback=lambda p: progress_bar.progress(p), # Lambda para pasar el valor de progreso
            max_workers=max_workers # Pasar el valor del slider
        )
        progress_bar.empty() # Opcional: limpiar la barra de progreso al finalizar

    st.success("✅ Elevaciones obtenidas")

    # --- Preparar datos para visualización ---
    # Filtrar puntos_con_elevacion por la ventana visible alrededor de km_actual
    kms_interp_arr = np.array([p.km for p in puntos_con_elevacion])
    elevs_interp_arr = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elevacion], dtype=float)

    mask_visible = (kms_interp_arr >= km_actual - ventana_km) & (kms_interp_arr <= km_actual + ventana_km)
    kms_vis = kms_interp_arr[mask_visible]
    elevs_vis = elevs_interp_arr[mask_visible]
    # Filtrar los puntos con elevación también para calcular la pendiente solo sobre los visibles
    puntos_visibles_para_pendiente = [puntos_con_elevacion[i] for i in range(len(puntos_con_elevacion)) if mask_visible[i]]


    # Calcular pendiente solo para los puntos visibles, usando el slider window_length
    if len(kms_vis) > 1: # Asegurarse de tener al menos 2 puntos para la pendiente
        # Necesitamos los arrays completos de kms y elevs para calcular la pendiente sobre el rango visible
        # calcular_pendiente_suavizada espera arrays de numpy
        pendientes_vis = calcular_pendiente_suavizada(kms_vis, elevs_vis, window_length=window_length)
    else:
        pendientes_vis = np.array([]) # Array vacío si no hay puntos visibles

    # --- Generar Gráfico ---
    st.subheader("📊 Perfil Altimétrico Visible")

    if len(kms_vis) > 1:
        # El Plotly figure ahora se genera en script2 y se retorna
        fig = go.Figure() # Crear una nueva figura solo con los datos visibles

        # Asegurarse de que los hover texts y pendientes coincidan con los puntos visibles
        # Esto requiere re-generar hover texts o filtrar los originales si es más eficiente.
        # Por simplicidad, generamos hover texts solo para los puntos visibles.
        hover_texts_vis = []
        for i, p_vis in enumerate(puntos_visibles_para_pendiente):
             km_text = f"<b>Km: {p_vis.km:.3f}</b><br>"
             elev_text = f"Elev: {p_vis.elevation:.1f} m" if p_vis.elevation is not None and not np.isnan(p_vis.elevation) else "Elev: N/A"
             slope_text = f"Pendiente: {pendientes_vis[i]:+.1f} m/km" if i < len(pendientes_vis) and not np.isnan(pendientes_vis[i]) else "Pendiente: N/A"
             hover_texts_vis.append(f"{km_text}{elev_text}<br>{slope_text}")


        # Trazado de Elevación (Solo puntos visibles)
        fig.add_trace(go.Scattergl(
            x=kms_vis, y=elevs_vis, mode='lines', name='Perfil',
            line=dict(color='cyan', width=3),
            hoverinfo='text', text=hover_texts_vis, # Usar hover texts filtrados/regenerados
            yaxis='y1'
        ))

        # Marca de posición actual
        fig.add_shape(type='line', x0=km_actual, x1=km_actual,
                      y0=min(elevs_vis) if len(elevs_vis) > 0 and not np.isnan(min(elevs_vis)) else 0,
                      y1=max(elevs_vis) if len(elevs_vis) > 0 and not np.isnan(max(elevs_vis)) else 100, # Rango Y más seguro si elevs_vis está vacío/NaN
                      line=dict(color='red', width=3, dash='dot'),
                      name=f"Km actual: {km_actual:.3f}") # Nombre para hover si se habilita

        # Añadir etiquetas de pendiente para los puntos visibles
        # Esto puede saturar el gráfico, solo si es necesario o con menos puntos.
        # Puedes adaptarlo para mostrar solo en intervalos o si el zoom es suficiente.
        # Por ahora, mantenemos la lógica original de añadir anotaciones entre puntos.
        if len(kms_vis) > 1 and len(pendientes_vis) > 0:
            for i in range(1, len(kms_vis)):
                mid_km = (kms_vis[i] + kms_vis[i-1]) / 2
                mid_elev = (elevs_vis[i] + elevs_vis[i-1]) / 2
                if i < len(pendientes_vis) and not np.isnan(pendientes_vis[i]):
                    pendiente_display = pendientes_vis[i]
                    # Añadir anotación solo si el punto medio está dentro de la ventana X visible
                    if mid_km >= kms_vis.min() and mid_km <= kms_vis.max():
                        fig.add_annotation(x=mid_km, y=mid_elev, # Ajustar posición Y si es necesario
                                           text=f"{pendiente_display:+.1f}",
                                           showarrow=False, font=dict(size=12, color='yellow'), # Color diferente para contraste
                                           yanchor="bottom") # Anclar texto debajo del punto medio


        # --- Configuración del Layout del Gráfico ---
        tramo_titulo = f"{est_inicio} - {est_fin}"
        fig.update_layout(
            paper_bgcolor='black', # Fondo del área del papel
            plot_bgcolor='black',  # Fondo del área del gráfico
            font=dict(color='white', size=14), # Tamaño de fuente general más pequeño
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(title='Kilómetro', color='white', showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=False), # Grid más sutil
            yaxis=dict(title='Elevación (m)', color='white', showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=False),
            title=dict(text=f"{tramo_titulo} | Km actual: {km_actual:.3f}", x=0.5, xanchor='center', font=dict(size=18)),
            hovermode='x unified', # Hover unificado
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
            # Añadir una marca de agua sutil
            annotations=[
                dict(
                    text="Perfil LAL", # Tu marca de agua
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=60, color="rgba(255,255,255,0.1)"), # Color semi-transparente
                    textangle=-30,
                    opacity=0.5,
                    layer="below" # Detrás de los datos
                )
            ]
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)

    elif len(kms_vis) > 0:
         st.warning("Se cargó el tramo, pero solo hay un punto visible en el rango actual.")
    else:
        st.warning("No hay puntos interpolados ni datos visibles en el rango seleccionado.")

elif archivo_subido and not df_estaciones.empty:
     # Este caso maneja archivos que se cargaron pero no tienen las columnas correctas
    st.error("❌ El archivo cargado no contiene las columnas requeridas: Nombre, Km, Lat, Lon")

# Si df_estaciones está vacío y no se ha subido ningún archivo, no mostramos nada.
# Si df_estaciones está vacío después de intentar cargar un archivo (por error),
# el mensaje de error específico ya se habrá mostrado.