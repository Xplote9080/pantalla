import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import xml.etree.ElementTree as ET
import io

# Importar el módulo script2 completo
import script2

# Ahora puedes acceder a las funciones y variables de script2 usando script2.nombre
# Por ejemplo: script2.cargar_estaciones, script2._cache, etc.

st.set_page_config(page_title="Pantalla Cabina", layout="wide")
st.title("🚆 Vista estilo cabina - Perfil altimétrico en tiempo real")

st.markdown("""
Cargá un archivo **CSV** o **KML** con estaciones (Nombre, Km, Lat, Lon). El sistema mostrará el perfil altimétrico alrededor del kilómetro actual.

También podés elegir un subtramo del archivo y cuántos *workers* usar para consultar elevaciones. Más workers = más rápido, pero puede ser rechazado por la API (Error 429).
""")

# --- Inicializar Caché de Elevaciones (Usando Streamlit's caching) ---
# Usamos st.cache_resource para cargar el caché solo una vez por despliegue.
@st.cache_resource
def initialize_elevation_cache():
    """Inicializa y carga el caché de elevaciones usando la función del script2."""
    cache_dict = script2.load_cache_to_memory()
    return cache_dict

script2._cache = initialize_elevation_cache()


# --- Procesar KML ---
def procesar_kml(kml_file_object):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    try:
        kml_file_object.seek(0)
        tree = ET.parse(kml_file_object)
    except Exception as e:
         st.error(f"Error al parsear el archivo KML. Asegúrate de que sea un archivo KML válido: {e}")
         return pd.DataFrame()
    root = tree.getroot()
    placemarks = root.findall('.//kml:Placemark', ns)
    datos = []
    if not placemarks:
        st.warning("No se encontraron Placemarks en el archivo KML.")
        return pd.DataFrame()

    for pm in placemarks:
        name_tag = pm.find('kml:name', ns)
        point_tag = pm.find('.//kml:Point', ns)
        coord_tag = point_tag.find('kml:coordinates', ns) if point_tag is not None else None

        nombre = name_tag.text.strip() if name_tag is not None and name_tag.text else ""
        km = np.nan
        lat = np.nan
        lon = np.nan

        if nombre:
            if ',' in nombre:
                partes_nombre = nombre.split(',', 1)
                nombre_limpio = partes_nombre[0].strip()
                km_str = partes_nombre[1].strip() if len(partes_nombre) > 1 else ""
                try:
                    km = float(km_str)
                except ValueError:
                    km = np.nan

            if coord_tag is not None and coord_tag.text:
                try:
                    coords_str = coord_tag.text.strip()
                    lon_str, lat_str, *alt_str = coords_str.split(',')
                    lat = float(lat_str)
                    lon = float(lon_str)
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                         lat = np.nan
                         lon = np.nan

                except ValueError as ve:
                     lat = np.nan
                     lon = np.nan
                except Exception as e:
                     lat = np.nan
                     lon = np.nan

        if nombre and not (np.isnan(lat) or np.isnan(lon)):
             datos.append({
                 'Nombre': nombre_limpio if 'nombre_limpio' in locals() else nombre,
                 'Km': km,
                 'Lat': lat,
                 'Lon': lon
             })

    return pd.DataFrame(datos)


# --- Cargar archivo ---
archivo_subido = st.file_uploader("📤 Subí tu archivo CSV o KML", type=["csv", "kml"])
df_estaciones = pd.DataFrame()

if archivo_subido:
    if archivo_subido.name.endswith(".csv"):
        try:
            df_estaciones = pd.read_csv(archivo_subido)
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            df_estaciones = pd.DataFrame()

    elif archivo_subido.name.endswith(".kml"):
        try:
            kml_bytes = archivo_subido.getvalue()
            kml_file_object = io.BytesIO(kml_bytes)
            df_estaciones = procesar_kml(kml_file_object)
        except Exception as e:
            st.error(f"Error al procesar el archivo KML: {e}")
            df_estaciones = pd.DataFrame()


# --- Validación y selección de subtramo ---
if not df_estaciones.empty and set(['Nombre', 'Km', 'Lat', 'Lon']).issubset(df_estaciones.columns):
    try:
        estaciones_cargadas = script2.cargar_estaciones(df_estaciones)
        df_estaciones_ordenadas = pd.DataFrame([s._asdict() for s in estaciones_cargadas])
    except ValueError as e:
        st.error(f"Error en los datos de las estaciones: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error inesperado al cargar las estaciones: {e}")
        st.stop()

    if df_estaciones_ordenadas.empty or len(df_estaciones_ordenadas) < 2:
        st.error("❌ No hay suficientes estaciones válidas con datos completos (Nombre, Km, Lat, Lon) después de la limpieza y validación.")
        st.stop()

    nombres_estaciones = df_estaciones_ordenadas["nombre"].tolist() # Usar "nombre" en minúscula

    st.subheader("📍 Selección del tramo a visualizar")

    default_index_fin = len(nombres_estaciones) - 1
    idx_est_inicio = st.selectbox("Estación inicial", options=range(len(nombres_estaciones)), format_func=lambda x: nombres_estaciones[x], index=0)
    idx_est_fin = st.selectbox("Estación final", options=range(len(nombres_estaciones)), format_func=lambda x: nombres_estaciones[x], index=default_index_fin)

    est_inicio = nombres_estaciones[idx_est_inicio]
    est_fin = nombres_estaciones[idx_est_fin]

    km_inicio = df_estaciones_ordenadas.loc[idx_est_inicio, "km"] # Usar "km" en minúscula
    km_fin = df_estaciones_ordenadas.loc[idx_est_fin, "km"]     # Usar "km" en minúscula

    if km_inicio >= km_fin:
        st.error("❌ La estación inicial debe tener un kilómetro menor que la final.")
        st.stop()

    estaciones_tramo_list = [s for s in estaciones_cargadas if s.km >= km_inicio and s.km <= km_fin]

    if not estaciones_tramo_list:
         st.error("❌ El tramo seleccionado no contiene estaciones válidas.")
         st.stop()

    st.success(f"✅ {len(estaciones_tramo_list)} estaciones seleccionadas entre {est_inicio} (Km {km_inicio:.3f}) y {est_fin} (Km {km_fin:.3f})")

    st.subheader("⚙️ Parámetros de visualización y procesamiento")

    intervalo = st.slider("Intervalo de interpolación (m)", 10, 500, 100, step=10,
                          help="Distancia en metros entre los puntos interpolados donde se consultará la elevación.")
    ventana_km = st.slider("Rango visible alrededor del Km actual (± Km)", 1, 20, 5,
                           help="Cuántos kilómetros antes y después del 'Kilómetro actual' se mostrarán en el gráfico.")

    min_km_tramo = estaciones_tramo_list[0].km
    max_km_tramo = estaciones_tramo_list[-1].km

    km_actual = st.slider("Kilómetro actual", float(min_km_tramo), float(max_km_tramo),
                          float((min_km_tramo + max_km_tramo)/2), step=0.1,
                          help="Simula la posición actual en la vía.")

    km_actual = max(float(min_km_tramo), min(float(max_km_tramo), km_actual))

    max_workers = st.slider("👷‍♂️ Número de workers para consultas a la API", 1, 10, 2,
                            help="Cuantos más workers, más rápido será, pero aumenta el riesgo de que la API rechace las consultas (Error 429: Too Many Requests). Reducir este valor ayuda a evitar el error 429.")

    window_length = st.slider("📏 Ventana de Suavizado para Pendiente (puntos)", 3, 51, 9, step=2,
                              help="Tamaño de la ventana (número de puntos) para el filtro Savitzky-Golay al calcular la pendiente. Un valor más alto suaviza más la pendiente. Debe ser un número impar.")

    st.info("Procesando datos... Esto puede tardar unos minutos dependiendo del tramo, el intervalo y la respuesta de la API.")

    # --- Procesamiento de Datos ---
    puntos_interp = script2.interpolar_puntos(estaciones_tramo_list, intervalo_m=intervalo)

    puntos_con_elevacion = []
    if puntos_interp:
        with st.spinner(f"Consultando elevaciones para {len(puntos_interp)} puntos (puede tomar tiempo)..."):
            progress_bar = st.progress(0)
            try:
                puntos_con_elevacion = script2.obtener_elevaciones_paralelo(
                    puntos_interp,
                    author="LAL",
                    progress_callback=lambda p: progress_bar.progress(p),
                    max_workers=max_workers
                )
                progress_bar.empty()
                st.success("✅ Elevaciones obtenidas")

            except Exception as e:
                progress_bar.empty()
                st.error(f"Error al obtener elevaciones: {e}. Intenta reducir el número de workers.")
                puntos_con_elevacion = []

    else:
        st.warning("No se generaron puntos para consultar elevación con el intervalo y tramo seleccionados.")
        puntos_con_elevacion = []

    # --- Preparar datos para visualización ---
    if puntos_con_elevacion:
        kms_interp_arr = np.array([p.km for p in puntos_con_elevacion])
        elevs_interp_arr = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elevacion], dtype=float)

        mask_visible = (kms_interp_arr >= km_actual - ventana_km) & (kms_interp_arr <= km_actual + ventana_km)
        kms_vis = kms_interp_arr[mask_visible]
        elevs_vis = elevs_interp_arr[mask_visible]
        puntos_visibles = [puntos_con_elevacion[i] for i in range(len(puntos_con_elevacion)) if mask_visible[i]]

        pendientes_vis = np.full_like(elevs_vis, np.nan, dtype=float)
        num_valid_vis = np.count_nonzero(~np.isnan(elevs_vis))

        if num_valid_vis >= window_length and window_length >= 3:
            try:
                pendientes_vis = script2.calcular_pendiente_suavizada(kms_vis, elevs_vis, window_length=window_length)
            except Exception as e:
                 st.error(f"Error al calcular la pendiente: {e}")
                 pendientes_vis = np.full_like(elevs_vis, np.nan, dtype=float)

        elif num_valid_vis > 1:
             st.warning(f"No hay suficientes puntos visibles válidos ({num_valid_vis}) para calcular la pendiente con una ventana de {window_length} (se requieren al menos {window_length}). Se mostrará el gráfico sin pendiente suavizada en el rango visible.")

        # --- Generar Gráfico ---
        st.subheader("📊 Perfil Altimétrico Visible")

        fig = None
        if len(kms_vis) > 1:
            try:
                fig = script2.graficar_html(
                    puntos_visibles,
                    estaciones_tramo_list,
                    titulo=f"{est_inicio} - {est_fin} | Km actual: {km_actual:.3f}",
                    slope_data_vis=pendientes_vis, # <-- CORRECCIÓN AQUÍ: Usar slope_data_vis
                    theme="dark",
                    colors="cyan,yellow",
                    watermark="LAL"
                )

                if fig:
                     y_min_vis = np.nanmin(elevs_vis) if not np.all(np.isnan(elevs_vis)) else 0
                     y_max_vis = np.nanmax(elevs_vis) if not np.all(np.isnan(elevs_vis)) else 100

                     if not np.isfinite(y_min_vis): y_min_vis = 0
                     if not np.isfinite(y_max_vis): y_max_vis = 100
                     if y_min_vis == y_max_vis:
                          y_min_vis, y_max_vis = y_min_vis - 10, y_max_vis + 10

                     fig.add_shape(type='line', x0=km_actual, x1=km_actual,
                                   y0=y_min_vis, y1=y_max_vis,
                                   line=dict(color='red', width=3, dash='dot'),
                                   name=f"Km actual: {km_actual:.3f}",
                                   xref='x', yref='y')

                     if len(kms_vis) > 1 and len(pendientes_vis) > 0:
                         valid_slope_indices = ~np.isnan(pendientes_vis)
                         kms_vis_valid_slope = kms_vis[valid_slope_indices]
                         elevs_vis_valid_slope = elevs_vis[valid_slope_indices]
                         pendientes_vis_valid = pendientes_vis[valid_slope_indices]

                         if len(kms_vis_valid_slope) > 0:
                             for i in range(len(kms_vis_valid_slope)):
                                 fig.add_annotation(x=kms_vis_valid_slope[i],
                                                    y=elevs_vis_valid_slope[i] + 5,
                                                    text=f"{pendientes_vis_valid[i]:+.1f}",
                                                    showarrow=False, font=dict(size=12, color='yellow'),
                                                    yanchor="bottom")

                     st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                 st.error(f"Error al generar el gráfico: {e}")
                 fig = None

        elif len(kms_vis) > 0:
             st.warning("Se cargó el tramo, pero solo hay un punto visible en el rango actual para graficar una línea.")
        else:
            st.warning("No hay puntos interpolados ni datos visibles en el rango seleccionado para graficar.")

        # --- Sección de Exportación ---
        st.subheader("💾 Exportar Datos y Gráfico")

        csv_buffer = io.StringIO()
        try:
            pendientes_completas = np.full_like(elevs_interp_arr, np.nan, dtype=float)
            num_valid_total = np.count_nonzero(~np.isnan(elevs_interp_arr))
            if num_valid_total >= window_length and window_length >= 3:
                 try:
                    pendientes_completas = script2.calcular_pendiente_suavizada(kms_interp_arr, elevs_interp_arr, window_length=window_length)
                 except Exception as e:
                    st.warning(f"Error al calcular pendientes para la exportación CSV completa: {e}. La columna de pendiente en el CSV podría tener NaNs.")
                    pendientes_completas = np.full_like(elevs_interp_arr, np.nan, dtype=float)
            elif num_valid_total > 1:
                 st.warning(f"No hay suficientes puntos interpolados totales válidos ({num_valid_total}) para calcular la pendiente con una ventana de {window_length} (se requieren al menos {window_length}). La columna de pendiente en el CSV estará vacía.")

            script2.exportar_csv(puntos_con_elevacion, pendientes_completas, csv_buffer, author="LAL")
            st.download_button(
                label="💾 Descargar Datos CSV (Todos los puntos)",
                data=csv_buffer.getvalue(),
                file_name=f"perfil_altimetrico_completo_{est_inicio.replace(' ','_')}_{est_fin.replace(' ','_')}.csv",
                mime="text/csv"
            )
        except Exception as e:
             st.error(f"Error al generar CSV: {e}")

        kml_buffer = io.BytesIO()
        try:
            script2.exportar_kml(puntos_con_elevacion, estaciones_tramo_list, kml_buffer, author="LAL")
            st.download_button(
                label="💾 Descargar KML (Estaciones)",
                data=kml_buffer.getvalue(),
                file_name=f"estaciones_perfil_{est_inicio.replace(' ','_')}_{est_fin.replace(' ','_')}.kml",
                mime="application/vnd.google-earth.kml+xml"
            )
        except Exception as e:
             st.error(f"Error al generar KML: {e}")

        geojson_buffer = io.StringIO()
        try:
            script2.exportar_geojson(puntos_con_elevacion, estaciones_tramo_list, geojson_buffer, author="LAL")
            st.download_button(
                label="💾 Descargar GeoJSON (Puntos y Estaciones)",
                data=geojson_buffer.getvalue(),
                file_name=f"perfil_altimetrico_datos_{est_inicio.replace(' ','_')}_{est_fin.replace(' ','_')}.geojson",
                mime="application/geo+json"
            )
        except Exception as e:
             st.error(f"Error al generar GeoJSON: {e}")

        if fig:
            try:
                pdf_buffer = io.BytesIO()
                script2.exportar_pdf(fig, pdf_buffer)
                st.download_button(
                    label="💾 Descargar Gráfico PDF",
                    data=pdf_buffer.getvalue(),
                    file_name=f"perfil_altimetrico_grafico_{est_inicio.replace(' ','_')}_{est_fin.replace(' ','_')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                 st.error(f"Error al generar PDF del gráfico: {e}. Asegúrate de que la librería 'kaleido' esté instalada y funcionando en tu entorno Streamlit Cloud.")

    elif archivo_subido and df_estaciones.empty:
         pass

if not archivo_subido:
     st.info("⬆️ Subí un archivo CSV o KML para empezar.")
