import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import xml.etree.ElementTree as ET
import io
import logging # Importar logging para mensajes más detallados

# Configuración básica de logging para ver mensajes en Streamlit Cloud logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
# Modificada para aceptar un objeto archivo (como io.BytesIO) y parsear Nombre/KM correctamente
def procesar_kml(kml_file_object):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    datos = []
    try:
        kml_file_object.seek(0)
        tree = ET.parse(kml_file_object)
        root = tree.getroot()
        placemarks = root.findall('.//kml:Placemark', ns)

        if not placemarks:
            st.warning("No se encontraron Placemarks en el archivo KML.")
            return pd.DataFrame()

        logging.info(f"Encontrados {len(placemarks)} Placemarks en el KML.")

        for i, pm in enumerate(placemarks):
            nombre_tag = pm.find('kml:name', ns)
            punto_tag = pm.find('.//kml:Point', ns)
            coords_tag = punto_tag.find('kml:coordinates', ns) if punto_tag is not None else None

            nombre_completo_str = nombre_tag.text.strip() if nombre_tag is not None and nombre_tag.text else ""
            lat = np.nan
            lon = np.nan
            parsed_name = ""
            parsed_km = np.nan

            # --- Parsear Nombre y KM ---
            if nombre_completo_str:
                try:
                    # Intentar dividir por " KM " para separar nombre y KM
                    km_split = nombre_completo_str.split(" KM ")
                    if len(km_split) > 1:
                        # Si se pudo dividir, la parte antes de " KM " es el nombre, la última es el KM
                        parsed_name = " KM ".join(km_split[:-1]).strip()
                        km_value_str = km_split[-1].strip()

                        # Limpiar y convertir el string del KM a float
                        # Eliminar caracteres no numéricos comunes al final si existen (ej. '?')
                        km_value_str_cleaned = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', km_value_str))
                        try:
                             parsed_km = float(km_value_str_cleaned)
                        except ValueError:
                            logging.warning(f"Placemark {i+1}: No se pudo convertir '{km_value_str_cleaned}' (obtenido de '{km_value_str}') a número KM desde el nombre '{nombre_completo_str}'. KM será NaN.")
                            parsed_km = np.nan # Asegurar que sea NaN si falla conversión
                    else:
                        # Si " KM " no se encuentra, el nombre completo es el nombre, KM es NaN
                        parsed_name = nombre_completo_str.strip()
                        logging.warning(f"Placemark {i+1}: No se encontró ' KM ' en el nombre '{nombre_completo_str}'. KM será NaN.")
                        parsed_km = np.nan # KM es NaN por defecto

                except Exception as e:
                     logging.error(f"Placemark {i+1}: Error inesperado al parsear Nombre/KM de '{nombre_completo_str}': {e}. KM será NaN.")
                     parsed_name = nombre_completo_str.strip() # Usar el nombre completo como fallback
                     parsed_km = np.nan # KM es NaN por defecto


            # --- Parsear Coordenadas ---
            if coords_tag is not None and coords_tag.text:
                try:
                    coords_str = coords_tag.text.strip()
                    # Las coordenadas en KML suelen ser lon,lat,alt (separadas por coma)
                    # Asegurarse de manejar comas como separador decimal si es el caso en la fuente original
                    # Pero KML estándar usa punto para decimal. Asumimos KML estándar lon,lat,alt con punto decimal.
                    partes_coord = coords_str.split(',')
                    if len(partes_coord) >= 2:
                        # Convertir lon y lat a float. KML usa punto decimal.
                        lon = float(partes_coord[0].strip())
                        lat = float(partes_coord[1].strip())

                        # Validar rangos de lat/lon
                        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                             logging.warning(f"Placemark {i+1}: Coordenadas fuera de rango para '{nombre_completo_str}': Lat={lat}, Lon={lon}. Lat/Lon será NaN.")
                             lat = np.nan # Marcar como inválido
                             lon = np.nan
                    else:
                         logging.warning(f"Placemark {i+1}: Formato de coordenadas inesperado '{coords_str}' para '{nombre_completo_str}'. Lat/Lon será NaN.")
                         lat = np.nan
                         lon = np.nan

                except ValueError as ve:
                     logging.warning(f"Placemark {i+1}: Error al convertir coordenadas a número para '{nombre_completo_str}' (coords: '{coords_str}'): {ve}. Lat/Lon será NaN.")
                     lat = np.nan
                     lon = np.nan
                except Exception as e:
                     logging.error(f"Placemark {i+1}: Error inesperado al procesar coordenadas para '{nombre_completo_str}' (coords: '{coords_str}'): {e}. Lat/Lon será NaN.")
                     lat = np.nan
                     lon = np.nan

            # --- Añadir a la lista de datos ---
            # Solo añadimos si se pudo obtener un nombre (aunque sea el nombre completo sin KM parseado)
            # y si las coordenadas Lat/Lon son válidas (no NaN).
            # El KM puede ser NaN si no se pudo parsear.
            if parsed_name and not (np.isnan(lat) or np.isnan(lon)):
                 datos.append({
                     'Nombre': parsed_name, # Usar el nombre parseado
                     'Km': parsed_km,      # Usar el KM parseado (puede ser NaN)
                     'Lat': lat,
                     'Lon': lon
                 })
            else:
                 if nombre_completo_str: # Si había un nombre original, loguear por qué se saltó
                      logging.warning(f"Placemark {i+1}: Saltando entrada KML para '{nombre_completo_str}' debido a nombre vacío o coordenadas inválidas/faltantes.")


        logging.info(f"Procesamiento KML finalizado. Encontrados {len(datos)} puntos con nombre y coordenadas válidas.")
        return pd.DataFrame(datos)

    except Exception as e:
        st.error(f"Error general al procesar el archivo KML: {e}")
        logging.error(f"Error general al procesar el archivo KML: {e}", exc_info=True)
        return pd.DataFrame() # Retornar DataFrame vacío en caso de error


# --- Cargar archivo ---
archivo_subido = st.file_uploader("📤 Subí tu archivo CSV o KML", type=["csv", "kml"])
df_estaciones = pd.DataFrame()

if archivo_subido:
    if archivo_subido.name.endswith(".csv"):
        try:
            # pandas read_csv puede leer directamente del objeto UploadedFile
            df_estaciones = pd.read_csv(archivo_subido)
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            logging.error(f"Error al leer el archivo CSV: {e}", exc_info=True)
            df_estaciones = pd.DataFrame()

    elif archivo_subido.name.endswith(".kml"):
        try:
            # Leer el contenido del archivo subido a un buffer en memoria (BytesIO)
            kml_bytes = archivo_subido.getvalue() # getvalue() obtiene los bytes del archivo
            kml_file_object = io.BytesIO(kml_bytes)
            # Pasar el objeto BytesIO a procesar_kml
            df_estaciones = procesar_kml(kml_file_object)
        except Exception as e:
            # Errores más específicos se loguean dentro de procesar_kml
            st.error(f"Error al procesar el archivo KML.")
            logging.error(f"Error al procesar el archivo KML: {e}", exc_info=True)
            df_estaciones = pd.DataFrame()


# --- Validación y selección de subtramo ---
# Verificar si el DataFrame no está vacío y tiene las columnas requeridas antes de cargarlas
# La función cargar_estaciones también valida y limpia, pero esta es una pre-validación rápida.
# Ahora, después de procesar KML, el DataFrame puede tener 'Km' como NaN. cargar_estaciones
# eliminará las filas con Km=NaN.
if not df_estaciones.empty and set(['Nombre', 'Km', 'Lat', 'Lon']).issubset(df_estaciones.columns):
    # Cargar y validar estaciones usando la función de script2
    # Esta función ya limpia, valida y ordena, y retorna una lista de namedtuples
    try:
        # cargar_estaciones eliminará las filas donde Km es NaN.
        estaciones_cargadas = script2.cargar_estaciones(df_estaciones)
        # Convertir la lista de namedtuples a DataFrame para usar con selectbox/loc
        # Las columnas serán 'nombre', 'km', 'lat', 'lon' (en minúscula)
        df_estaciones_ordenadas = pd.DataFrame([s._asdict() for s in estaciones_cargadas])
    except ValueError as e:
        st.error(f"Error en los datos de las estaciones: {e}")
        logging.error(f"Error en los datos de las estaciones: {e}", exc_info=True)
        st.stop() # Detener si los datos de entrada son inválidos
    except Exception as e:
        st.error(f"Error inesperado al cargar las estaciones: {e}")
        logging.error(f"Error inesperado al cargar las estaciones: {e}", exc_info=True)
        st.stop()


    if df_estaciones_ordenadas.empty or len(df_estaciones_ordenadas) < 2:
        st.error("❌ No hay suficientes estaciones válidas con datos completos (Nombre, Km, Lat, Lon) después de la limpieza y validación. Asegúrate de que los nombres en tu KML contengan el patrón ' KM ##.#' y que las coordenadas sean válidas.")
        st.stop()

    # Ahora acceder a la columna 'nombre' en minúscula
    nombres_estaciones = df_estaciones_ordenadas["nombre"].tolist()

    st.subheader("📍 Selección del tramo a visualizar")

    # Usar índices para los selectbox es más seguro
    # Encontrar el índice de la última estación para el valor por defecto del selectbox 'fin'
    default_index_fin = len(nombres_estaciones) - 1
    idx_est_inicio = st.selectbox("Estación inicial", options=range(len(nombres_estaciones)), format_func=lambda x: nombres_estaciones[x], index=0)
    idx_est_fin = st.selectbox("Estación final", options=range(len(nombres_estaciones)), format_func=lambda x: nombres_estaciones[x], index=default_index_fin)

    est_inicio = nombres_estaciones[idx_est_inicio]
    est_fin = nombres_estaciones[idx_est_fin]

    # Acceder a la columna 'km' en minúscula
    km_inicio = df_estaciones_ordenadas.loc[idx_est_inicio, "km"]
    km_fin = df_estaciones_ordenadas.loc[idx_est_fin, "km"]

    if km_inicio >= km_fin:
        st.error("❌ La estación inicial debe tener un kilómetro menor que la final.")
        st.stop()

    # Filtrar las estaciones cargadas (ya validadas y ordenadas) por el tramo seleccionado en base a su km
    # Usar la lista original de namedtuples 'estaciones_cargadas' que está ordenada por km
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
                logging.error(f"Error al obtener elevaciones: {e}", exc_info=True)
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
                 logging.error(f"Error al calcular la pendiente: {e}", exc_info=True)
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
                    slope_data_vis=pendientes_vis,
                    theme="dark",
                    colors="cyan,yellow",
                    watermark="Perfil LAL 2025"
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
                 logging.error(f"Error al generar el gráfico: {e}", exc_info=True)
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
                    logging.error(f"Error al calcular pendientes para la exportación CSV completa: {e}", exc_info=True)
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
             logging.error(f"Error al generar CSV: {e}", exc_info=True)

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
             logging.error(f"Error al generar KML: {e}", exc_info=True)


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
             logging.error(f"Error al generar GeoJSON: {e}", exc_info=True)

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
                 logging.error(f"Error al generar PDF del gráfico: {e}", exc_info=True)


    elif archivo_subido and df_estaciones.empty:
         pass

if not archivo_subido:
     st.info("⬆️ Subí un archivo CSV o KML para empezar.")
