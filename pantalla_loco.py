import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
# import tempfile # Ya no es necesario para KML upload, pero puede ser para exportaciones si las funciones no aceptan buffers
import os # Aún necesario para manejar tempfiles si se usan para exportación
import xml.etree.ElementTree as ET
import io # Importar io para manejar buffers

# Importar todas las funciones necesarias, incluyendo las de exportación
from script2 import (
    cargar_estaciones,
    interpolar_puntos,
    obtener_elevaciones_paralelo,
    calcular_pendiente_suavizada,
    load_cache_to_memory, # Importar la función de carga de caché
    _cache, # Importar la variable global del caché (para asignarla)
    exportar_kml, # Importar funciones de exportación
    exportar_geojson,
    exportar_csv,
    exportar_pdf
)

st.set_page_config(page_title="Pantalla Cabina", layout="wide")
st.title("🚆 Vista estilo cabina - Perfil altimétrico en tiempo real")

st.markdown("""
Cargá un archivo **CSV** o **KML** con estaciones (Nombre, Km, Lat, Lon). El sistema mostrará el perfil altimétrico alrededor del kilómetro actual.

También podés elegir un subtramo del archivo y cuántos *workers* usar para consultar elevaciones. Más workers = más rápido, pero puede ser rechazado por la API (Error 429).
""")

# --- Inicializar Caché de Elevaciones (Usando Streamlit's caching) ---
# Usamos st.cache_resource para cargar el caché solo una vez por despliegue.
# Esto llama a load_cache_to_memory() en script2.py la primera vez que se ejecuta
# y reutiliza el resultado en ejecuciones posteriores.
# Importante: st.cache_resource se basa en la identidad del objeto retornado.
# Retornar y asignar el dict del caché asegura que script2._cache apunte al mismo objeto
# en todas las ejecuciones de Streamlit.
@st.cache_resource
def initialize_elevation_cache():
    """Inicializa y carga el caché de elevaciones usando la función del script2."""
    # La función load_cache_to_memory carga desde el archivo si existe y retorna el dict.
    cache_dict = load_cache_to_memory()
    return cache_dict

# Llamar para inicializar el caché y asignarlo a la variable global en script2
# Esto es crucial para que las funciones de script2 (_load_elevation_from_cache, _save_elevation_to_cache)
# operen sobre la misma instancia del caché cargado por Streamlit.
script2._cache = initialize_elevation_cache()


# --- Procesar KML ---
# Modificada para aceptar un objeto archivo (como io.BytesIO)
def procesar_kml(kml_file_object):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    # ET.parse puede leer de un objeto tipo file
    try:
        tree = ET.parse(kml_file_object)
    except Exception as e:
         st.error(f"Error al parsear el archivo KML. Asegúrate de que sea un archivo KML válido: {e}")
         return pd.DataFrame() # Retornar DataFrame vacío en caso de error
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

        if nombre: # Solo procesar si hay nombre
            # Intentar parsear Nombre y Km del texto
            if ',' in nombre:
                partes_nombre = nombre.split(',', 1) # Split solo en la primera coma
                nombre_limpio = partes_nombre[0].strip()
                km_str = partes_nombre[1].strip() if len(partes_nombre) > 1 else ""
                try:
                    km = float(km_str)
                except ValueError:
                    st.warning(f"No se pudo parsear KM de '{km_str}' en '{nombre}'. Fila marcada con KM inválido.")
                    km = np.nan # Marcar como inválido si falla

            # Procesar coordenadas si existen
            if coord_tag is not None and coord_tag.text:
                try:
                    coords_str = coord_tag.text.strip()
                    # Las coordenadas en KML suelen ser lon, lat, alt
                    lon_str, lat_str, *alt_str = coords_str.split(',')
                    lat = float(lat_str)
                    lon = float(lon_str)
                    # alt = float(alt_str[0]) if alt_str else np.nan # Opcional: leer altitud si existe
                    # Validar rangos de lat/lon
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                         st.warning(f"Coordenadas fuera de rango para '{nombre}': Lat={lat}, Lon={lon}. Fila marcada con Lat/Lon inválido.")
                         lat = np.nan # Marcar como inválido
                         lon = np.nan


                except ValueError as ve:
                     st.warning(f"Error al parsear coordenadas '{coords_str}' para '{nombre}': {ve}. Fila marcada con Lat/Lon inválido.")
                     lat = np.nan # Marcar como inválido
                     lon = np.nan
                except Exception as e:
                     st.warning(f"Error inesperado al procesar coordenadas para '{nombre}': {e}. Fila marcada con Lat/Lon inválido.")
                     lat = np.nan # Marcar como inválido
                     lon = np.nan


        # Añadir datos si al menos el Nombre es válido
        if nombre and not (np.isnan(km) or np.isnan(lat) or np.isnan(lon)):
            datos.append({
                'Nombre': nombre_limpio if 'nombre_limpio' in locals() else nombre, # Usar limpio si se hizo split
                'Km': km,
                'Lat': lat,
                'Lon': lon
            })
        else:
             if nombre: # Si al menos hay nombre, loguear por qué se saltó
                st.warning(f"Saltando entrada KML para '{nombre}' debido a datos faltantes o inválidos (Km, Lat, Lon).")
             # else: entrada completamente vacía, ignorar silenciosamente

    return pd.DataFrame(datos)


# --- Cargar archivo ---
archivo_subido = st.file_uploader("📤 Subí tu archivo CSV o KML", type=["csv", "kml"])
df_estaciones = pd.DataFrame() # Inicializar vacío

if archivo_subido:
    if archivo_subido.name.endswith(".csv"):
        try:
            # pandas read_csv puede leer directamente del objeto UploadedFile
            df_estaciones = pd.read_csv(archivo_subido)
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            df_estaciones = pd.DataFrame() # Asegurar que esté vacío en caso de error

    elif archivo_subido.name.endswith(".kml"):
        try:
            # Leer el contenido del archivo subido a un buffer en memoria (BytesIO)
            kml_bytes = archivo_subido.getvalue() # getvalue() obtiene los bytes del archivo
            kml_file_object = io.BytesIO(kml_bytes)
            # Pasar el objeto BytesIO a procesar_kml
            df_estaciones = procesar_kml(kml_file_object)
        except Exception as e:
            st.error(f"Error al procesar el archivo KML: {e}")
            df_estaciones = pd.DataFrame() # Asegurar que esté vacío en caso de error
        # Ya no se necesita cleanup de archivo temporal aquí


# --- Validación y selección de subtramo ---
# Verificar si el DataFrame no está vacío y tiene las columnas requeridas
if not df_estaciones.empty and set(['Nombre', 'Km', 'Lat', 'Lon']).issubset(df_estaciones.columns):
    # Cargar y validar estaciones usando la función de script2
    # Esta función ya limpia, valida y ordena
    try:
        estaciones_cargadas = cargar_estaciones(df_estaciones)
        df_estaciones_ordenadas = pd.DataFrame([s._asdict() for s in estaciones_cargadas]) # Convertir de nuevo a DF para selectbox
    except ValueError as e:
        st.error(f"Error en los datos de las estaciones: {e}")
        st.stop() # Detener si los datos de entrada son inválidos
    except Exception as e:
        st.error(f"Error inesperado al cargar las estaciones: {e}")
        st.stop()


    if df_estaciones_ordenadas.empty or len(df_estaciones_ordenadas) < 2:
        st.error("❌ No hay suficientes estaciones válidas con datos completos (Nombre, Km, Lat, Lon) después de la limpieza y validación.")
        st.stop()

    nombres_estaciones = df_estaciones_ordenadas["Nombre"].tolist()

    st.subheader("📍 Selección del tramo a visualizar")

    # Usar índices para los selectbox es más seguro
    # Encontrar el índice de la última estación para el valor por defecto del selectbox 'fin'
    default_index_fin = len(nombres_estaciones) - 1
    idx_est_inicio = st.selectbox("Estación inicial", options=range(len(nombres_estaciones)), format_func=lambda x: nombres_estaciones[x], index=0)
    idx_est_fin = st.selectbox("Estación final", options=range(len(nombres_estaciones)), format_func=lambda x: nombres_estaciones[x], index=default_index_fin)

    est_inicio = nombres_estaciones[idx_est_inicio]
    est_fin = nombres_estaciones[idx_est_fin]

    km_inicio = df_estaciones_ordenadas.loc[idx_est_inicio, "Km"]
    km_fin = df_estaciones_ordenadas.loc[idx_est_fin, "Km"]

    if km_inicio >= km_fin:
        st.error("❌ La estación inicial debe tener un kilómetro menor que la final.")
        st.stop()

    # Filtrar las estaciones cargadas (ya validadas y ordenadas) por el tramo seleccionado
    estaciones_tramo_list = [s for s in estaciones_cargadas if s.km >= km_inicio and s.km <= km_fin]


    if not estaciones_tramo_list:
         st.error("❌ El tramo seleccionado no contiene estaciones válidas.")
         st.stop()

    st.success(f"✅ {len(estaciones_tramo_list)} estaciones seleccionadas entre {est_inicio} (Km {km_inicio:.3f}) y {est_fin} (Km {km_fin:.3f})")

    st.subheader("⚙️ Parámetros de visualización y procesamiento")

    # Controles para parámetros
    intervalo = st.slider("Intervalo de interpolación (m)", 10, 500, 100, step=10,
                          help="Distancia en metros entre los puntos interpolados donde se consultará la elevación.")
    ventana_km = st.slider("Rango visible alrededor del Km actual (± Km)", 1, 20, 5,
                           help="Cuántos kilómetros antes y después del 'Kilómetro actual' se mostrarán en el gráfico.")
    # Rango del slider de Km actual basado en el tramo seleccionado
    min_km_tramo = estaciones_tramo_list[0].km
    max_km_tramo = estaciones_tramo_list[-1].km

    km_actual = st.slider("Kilómetro actual", float(min_km_tramo), float(max_km_tramo),
                          float((min_km_tramo + max_km_tramo)/2), step=0.1,
                          help="Simula la posición actual en la vía.")

    # Asegurarse de que km_actual esté dentro del rango del tramo seleccionado
    km_actual = max(float(min_km_tramo), min(float(max_km_tramo), km_actual))


    max_workers = st.slider("👷‍♂️ Número de workers para consultas a la API", 1, 10, 2,
                            help="Cuantos más workers, más rápido será, pero aumenta el riesgo de que la API rechace las consultas (Error 429: Too Many Requests). Reducir este valor ayuda a evitar el error 429.")

    # Slider para la ventana de suavizado de la pendiente
    # Asegurar que sea impar y al menos 3
    window_length = st.slider("📏 Ventana de Suavizado para Pendiente (puntos)", 3, 51, 9, step=2,
                              help="Tamaño de la ventana (número de puntos) para el filtro Savitzky-Golay al calcular la pendiente. Un valor más alto suaviza más la pendiente. Debe ser un número impar.")


    st.info("Procesando datos... Esto puede tardar unos minutos dependiendo del tramo, el intervalo y la respuesta de la API.")

    # --- Procesamiento de Datos ---
    # La interpolación se hace sobre la lista de estaciones del tramo
    puntos_interp = interpolar_puntos(estaciones_tramo_list, intervalo_m=intervalo)


    # Usar un spinner y progress bar para mostrar que se está procesando
    if puntos_interp:
        with st.spinner(f"Consultando elevaciones para {len(puntos_interp)} puntos (puede tomar tiempo)..."):
            progress_bar = st.progress(0)
            try:
                puntos_con_elevacion = obtener_elevaciones_paralelo(
                    puntos_interp,
                    author="LAL", # Pasa tu autor
                    progress_callback=lambda p: progress_bar.progress(p), # Lambda para pasar el valor de progreso
                    max_workers=max_workers # Pasar el valor del slider
                )
                progress_bar.empty() # Opcional: limpiar la barra de progreso al finalizar
                st.success("✅ Elevaciones obtenidas")

            except Exception as e:
                progress_bar.empty()
                st.error(f"Error al obtener elevaciones: {e}. Intenta reducir el número de workers.")
                puntos_con_elevacion = [] # Asegurarse de que esté vacío si falla

    else:
        st.warning("No se generaron puntos para consultar elevación con el intervalo y tramo seleccionados.")
        puntos_con_elevacion = []


    # --- Preparar datos para visualización ---
    if puntos_con_elevacion:
        kms_interp_arr = np.array([p.km for p in puntos_con_elevacion])
        # Convertir la lista de puntos con elevación (que puede tener None) a array numpy con NaN
        elevs_interp_arr = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elevacion], dtype=float)

        # Filtrar puntos_con_elevacion por la ventana visible alrededor de km_actual
        mask_visible = (kms_interp_arr >= km_actual - ventana_km) & (kms_interp_arr <= km_actual + ventana_km)
        kms_vis = kms_interp_arr[mask_visible]
        elevs_vis = elevs_interp_arr[mask_visible]
        # Filtrar los puntos con elevación también para calcular la pendiente solo sobre los visibles
        # Necesitamos los objetos Point completos para los hover texts más adelante
        puntos_visibles = [puntos_con_elevacion[i] for i in range(len(puntos_con_elevacion)) if mask_visible[i]]


        # Calcular pendiente solo para los puntos visibles, usando el slider window_length
        if len(kms_vis) >= window_length: # Asegurarse de tener suficientes puntos válidos para el filtro
            # calcular_pendiente_suavizada espera arrays de numpy
            pendientes_vis = calcular_pendiente_suavizada(kms_vis, elevs_vis, window_length=window_length)
        else:
            if len(kms_vis) > 1: # Si hay puntos pero no suficientes para el filtro
                 st.warning(f"No hay suficientes puntos visibles ({len(kms_vis)}) para calcular la pendiente con una ventana de {window_length}. Se mostrará el gráfico sin pendiente.")
            pendientes_vis = np.full_like(elevs_vis, np.nan, dtype=float) # Array de NaNs si no se puede calcular

        # --- Generar Gráfico ---
        st.subheader("📊 Perfil Altimétrico Visible")

        if len(kms_vis) > 1: # Asegurarse de tener al menos 2 puntos visibles para dibujar la línea
            # Generar la figura Plotly usando la función de script2
            fig = graficar_html(
                puntos_visibles, # Pasar solo los puntos visibles para el gráfico
                estaciones_tramo_list, # Pasar todas las estaciones del tramo para marcadores
                # archivo_html="", # No se guarda a archivo HTML
                titulo=f"{est_inicio} - {est_fin} | Km actual: {km_actual:.3f}",
                slope_data=pendientes_vis, # Pasar las pendientes calculadas para los puntos visibles
                theme="dark", # Usar tema oscuro
                colors="cyan,yellow", # Colores para elevación y pendiente
                watermark="Perfil LAL 2025" # Marca de agua
            )

            # Añadir la marca de posición actual al gráfico generado
            # Es mejor añadir esto aquí en la UI script
            if fig: # Asegurarse de que la figura se generó
                 # Calcular min/max Y del gráfico visible para la línea vertical
                 y_min_vis = np.nanmin(elevs_vis) if not np.all(np.isnan(elevs_vis)) else 0
                 y_max_vis = np.nanmax(elevs_vis) if not np.all(np.isnan(elevs_vis)) else 100

                 fig.add_shape(type='line', x0=km_actual, x1=km_actual,
                               y0=y_min_vis, y1=y_max_vis, # Usar min/max de las elevaciones visibles
                               line=dict(color='red', width=3, dash='dot'),
                               name=f"Km actual: {km_actual:.3f}", # Nombre para hover
                               xref='x', yref='y') # Referenciar a los ejes de datos


                 # Mostrar el gráfico en Streamlit
                 st.plotly_chart(fig, use_container_width=True)

                 # --- Sección de Exportación ---
                 st.subheader("💾 Exportar Datos y Gráfico")

                 # Botón para exportar CSV
                 csv_buffer = io.StringIO() # Buffer para datos de texto
                 try:
                     # exportar_csv ahora acepta el buffer directamente
                     exportar_csv(puntos_con_elevacion, np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elevacion], dtype=float), csv_buffer, author="LAL") # Usar todos los puntos con elevación
                     st.download_button(
                         label="💾 Descargar Datos CSV (Todos los puntos)",
                         data=csv_buffer.getvalue(),
                         file_name=f"perfil_altimetrico_completo_{est_inicio}_{est_fin}.csv",
                         mime="text/csv"
                     )
                 except Exception as e:
                      st.error(f"Error al generar CSV: {e}")

                 # Botón para exportar KML
                 kml_buffer = io.BytesIO() # Buffer para datos binarios
                 try:
                     # exportar_kml ahora acepta el buffer directamente (usando tempfile internamente)
                     exportar_kml(puntos_con_elevacion, estaciones_tramo_list, kml_buffer, author="LAL") # Exportar estaciones y puntos interpolados (si aplica en func)
                     st.download_button(
                         label="💾 Descargar KML (Estaciones)", # KML solo exporta estaciones con elevación interpolada más cercana
                         data=kml_buffer.getvalue(),
                         file_name=f"estaciones_perfil_{est_inicio}_{est_fin}.kml",
                         mime="application/vnd.google-earth.kml+xml"
                     )
                 except Exception as e:
                      st.error(f"Error al generar KML: {e}")


                 # Botón para exportar GeoJSON
                 geojson_buffer = io.StringIO() # Buffer para datos de texto
                 try:
                     # exportar_geojson ahora acepta el buffer directamente
                     exportar_geojson(puntos_con_elevacion, estaciones_tramo_list, geojson_buffer, author="LAL") # Exportar puntos interpolados y estaciones
                     st.download_button(
                         label="💾 Descargar GeoJSON (Puntos y Estaciones)",
                         data=geojson_buffer.getvalue(),
                         file_name=f"perfil_altimetrico_datos_{est_inicio}_{est_fin}.geojson",
                         mime="application/geo+json"
                     )
                 except Exception as e:
                      st.error(f"Error al generar GeoJSON: {e}")

                 # Botón para exportar PDF del gráfico
                 try:
                     # exportar_pdf ahora acepta el buffer directamente
                     pdf_buffer = io.BytesIO()
                     exportar_pdf(fig, pdf_buffer)
                     st.download_button(
                         label="💾 Descargar Gráfico PDF",
                         data=pdf_buffer.getvalue(),
                         file_name=f"perfil_altimetrico_grafico_{est_inicio}_{est_fin}.pdf",
                         mime="application/pdf"
                     )
                 except Exception as e:
                      st.error(f"Error al generar PDF del gráfico: {e}. Asegúrate de que la librería 'kaleido' esté instalada y funcionando en tu entorno Streamlit Cloud.")


        elif len(kms_vis) > 0:
             st.warning("Se cargó el tramo, pero solo hay un punto visible en el rango actual para graficar una línea.")
        else:
            st.warning("No hay puntos interpolados ni datos visibles en el rango seleccionado para graficar.")

    elif archivo_subido and not df_estaciones.empty:
         # Este caso maneja archivos que se cargaron pero no tienen las columnas correctas
         # Este error ya se maneja arriba al llamar a cargar_estaciones
         pass # No hacer nada aquí para evitar doble mensaje

elif archivo_subido and df_estaciones.empty:
     # Este caso ocurre si la carga inicial del archivo falló
     pass # El error específico ya se mostró al cargar el archivo


# Si df_estaciones está vacío y no se ha subido ningún archivo, no mostramos nada.
# Si df_estaciones está vacío después de intentar cargar un archivo (por error),
# el mensaje de error específico ya se habrá mostrado.