# -*- coding: utf-8 -*-
"""
Script para generar perfiles altimétricos a partir de datos de estaciones para vías férreas.
© 2025 LAL - Todos los derechos reservados.

Funcionalidades:
- Carga estaciones desde un DataFrame (anteriormente CSV).
- Interpola puntos usando CubicSpline.
- Obtiene elevaciones con la API de Open-Meteo (con caché en CSV y llamadas por lotes paralelas).
- Calcula pendientes suavizadas (m/km) con filtro Savitzky-Golay.
- Genera gráficos interactivos con Plotly.
- Exporta a PDF, KML, CSV y GeoJSON (adaptadas para posible uso con buffers en UI).
- Valida rangos de latitud, longitud y kilómetros.
- Incluye atribución: "Perfil altimétrico ferroviario - LAL 2025".
- Optimizado para Streamlit Cloud con progreso visual y manejo de archivos temporales.
"""

import csv
import requests
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import plotly.graph_objects as go
# from plotly.offline import plot # No necesario para Streamlit
import simplekml
from geojson import Point, Feature, FeatureCollection, dump
import logging
import os
import tempfile # Aún necesario para exportaciones KML si SimpleKML no soporta buffers directamente
import pandas as pd
from typing import List, Optional, NamedTuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, wait_random
import io # Importar io para manejar buffers

# --- Configuración y Constantes ---
ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_CACHE_FILE = "elevations_cache.csv"
DEFAULT_INTERVAL_METERS = 200
REQUEST_TIMEOUT_SECONDS = 15
MAX_API_WORKERS = 4 # Se controla desde Streamlit
MAX_BATCH_SIZE = 50 # Reducido para mitigar error 429
DEFAULT_ELEVATION_ON_ERROR = 0.0
DEFAULT_SMOOTH_WINDOW = 9 # Valor por defecto actualizado, pero se controla desde Streamlit
AUTHOR_ATTRIBUTION = "Perfil altimétrico ferroviario - LAL 2025"

# Configuración del Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Estructuras de Datos ---
class Station(NamedTuple):
    nombre: str
    km: float
    lat: float
    lon: float

class InterpolatedPoint(NamedTuple):
    km: float
    lat: float
    lon: float
    elevation: Optional[float] = None

# --- Caché en memoria ---
_cache = None

# COMENTARIO SOBRE ESCALABILIDAD DEL CACHÉ:
# El caché actual en memoria (_cache) y en disco (elevations_cache.csv) crece
# indefinidamente. Para aplicaciones de larga duración o que procesan grandes
# volúmenes de datos en diferentes áreas, esto podría causar problemas de memoria.
# Una mejora futura sería implementar un mecanismo de limpieza (ej. LRU)
# o usar una solución de caché más escalable (ej. base de datos).
# Por ahora, asumimos que el volumen de datos no es excesivamente grande.

def load_cache_to_memory() -> Dict[tuple[float, float], float]:
    """Carga el caché de elevaciones en memoria desde el CSV."""
    global _cache
    if _cache is not None:
        return _cache # Si ya está cargado en memoria (ej. por Streamlit cache), retornarlo
    _cache = {}
    if not os.path.exists(ELEVATION_CACHE_FILE):
        logging.info("Archivo de caché no encontrado. Se creará uno nuevo si se obtienen datos.")
        return _cache
    try:
        with open(ELEVATION_CACHE_FILE, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            # Saltar líneas de comentario/cabecera
            for row in reader:
                if row and not row[0].startswith('#') and row[0].lower() != 'latitude':
                    try:
                        lat, lon, elev = map(float, row[:3])
                        # Redondear para consistencia
                        _cache[(round(lat, 5), round(lon, 5))] = elev
                    except ValueError:
                        logging.warning(f"Fila inválida en caché CSV: {row}")
        logging.info(f"Cargadas {len(_cache)} entradas desde el caché.")
    except Exception as e:
        logging.warning(f"Error al leer caché CSV '{ELEVATION_CACHE_FILE}': {e}")
    return _cache

def _load_elevation_from_cache(lat: float, lon: float) -> Optional[float]:
    """Carga elevación desde el caché en memoria."""
    # Asegura que el caché esté cargado. En Streamlit, se manejará con st.cache_resource.
    # load_cache_to_memory() # No llamar aquí si se usa st.cache_resource en UI
    cache = _cache # Usar la variable global que st.cache_resource manejará
    if cache is None: # Seguridad si no se usa st.cache_resource correctamente
         cache = load_cache_to_memory()
    lat_r = round(lat, 5)
    lon_r = round(lon, 5)
    return cache.get((lat_r, lon_r))

def _save_elevation_to_cache(lat: float, lon: float, elevation: float, author: str = AUTHOR_ATTRIBUTION):
    """Guarda elevación en el archivo CSV y en memoria."""
    global _cache
    # Asegurarse de que el caché en memoria esté actualizado
    lat_r = round(lat, 5)
    lon_r = round(lon, 5)

    if _cache is not None and (lat_r, lon_r) in _cache:
        return # Evitar escribir duplicados si ya está en memoria

    # Si _cache es None, cargarlo (esto no debería pasar con st.cache_resource bien configurado)
    # pero lo dejamos como seguridad
    if _cache is None:
        load_cache_to_memory()

    _cache[(lat_r, lon_r)] = elevation # Actualizar caché en memoria


    try:
        file_exists = os.path.exists(ELEVATION_CACHE_FILE)
        # Abrir en modo 'a' (append) y 'x' para creación exclusiva si no existe.
        # Usamos 'a' simple y verificamos si está vacío para mayor compatibilidad.
        with open(ELEVATION_CACHE_FILE, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Escribir cabecera solo si el archivo es nuevo o está vacío
            if not file_exists or os.path.getsize(ELEVATION_CACHE_FILE) == 0:
                writer.writerow([f"# {author} - Caché de elevaciones para vías férreas"])
                writer.writerow(["latitude", "longitude", "elevation"])
            writer.writerow([lat_r, lon_r, elevation])
    except Exception as e:
        logging.warning(f"Error al guardar en caché CSV '{ELEVATION_CACHE_FILE}': {e}")

# --- Funciones principales (sin cambios mayores en lógica de cálculo) ---
def cargar_estaciones(df: pd.DataFrame) -> List[Station]:
    """Carga estaciones desde un DataFrame de Pandas."""
    try:
        df_cleaned = df.copy() # Trabajar en una copia
        df_cleaned.columns = [c.strip().capitalize() for c in df_cleaned.columns]
        required_columns = {'Nombre', 'Km', 'Lat', 'Lon'}
        if not required_columns.issubset(df_cleaned.columns):
            raise ValueError(f"El DataFrame debe contener las columnas: {required_columns}")

        # Limpieza y conversión a tipos numéricos, eliminando filas inválidas
        df_cleaned['Km'] = pd.to_numeric(df_cleaned['Km'], errors='coerce')
        df_cleaned['Lat'] = pd.to_numeric(df_cleaned['Lat'], errors='coerce')
        df_cleaned['Lon'] = pd.to_numeric(df_cleaned['Lon'], errors='coerce')
        df_cleaned.dropna(subset=['Nombre', 'Km', 'Lat', 'Lon'], inplace=True) # Nombre también por si acaso es None/NaN

        estaciones = []
        # Usamos itertuples() que suele ser más rápido que iterrows()
        for row in df_cleaned.itertuples(index=False):
             # Las conversiones ya se hicieron y validaron con dropna/coerce
             estaciones.append(Station(
                 nombre=str(row.Nombre).strip(),
                 km=row.Km,
                 lat=row.Lat,
                 lon=row.Lon
             ))

        if len(estaciones) < 2:
            raise ValueError("Se requieren al menos 2 estaciones válidas con datos completos (Nombre, Km, Lat, Lon).")

        # Verificar y asegurar orden creciente por Km
        estaciones_ordenadas = sorted(estaciones, key=lambda s: s.km)
        if not all(estaciones_ordenadas[i].km < estaciones_ordenadas[i+1].km for i in range(len(estaciones_ordenadas)-1)):
            # Si no son estrictamente crecientes, incluso después de ordenar
            # Podría haber kilómetros duplicados. Si los duplicados son aceptables pero deben procesarse,
            # se requeriría otra lógica (ej. promediar coordenadas, o manejar como puntos distintos).
            # Por ahora, mantenemos la estricta necesidad de Km crecientes para la interpolación spline.
            # Si hay kilómetros duplicados que impiden el orden estrictamente creciente, lanza error.
            kms_unicos = sorted(list(set(s.km for s in estaciones_ordenadas)))
            if len(kms_unicos) < len(estaciones_ordenadas):
                 raise ValueError("Los kilómetros deben ser estrictamente crecientes. Se encontraron kilómetros duplicados.")
            # Si los kilómetros únicos son igual al número de estaciones pero aún así no son estrictamente crecientes
            # (esto es matemáticamente imposible si ya están ordenados), entonces hay un error lógico.
            # Pero la verificación sorted() + all() debería cubrir el caso de no ser estrictamente creciente.

        logging.info(f"Cargadas {len(estaciones_ordenadas)} estaciones válidas desde el DataFrame")
        return estaciones_ordenadas # Retornar la lista ordenada

    except Exception as e:
        logging.error(f"Error al cargar estaciones desde DataFrame: {e}")
        raise # Propagar la excepción para que Streamlit la muestre


def interpolar_puntos(estaciones: List[Station], intervalo_m: int = DEFAULT_INTERVAL_METERS) -> List[InterpolatedPoint]:
    """Interpola puntos geográficos."""
    # La lógica de interpolación se mantiene igual, asumiendo estaciones ordenadas por Km.
    # (Ya se garantiza en cargar_estaciones)
    if len(estaciones) < 2:
        logging.warning("Mínimo 2 estaciones requeridas para interpolación")
        return []

    kms = np.array([s.km for s in estaciones])
    lats = np.array([s.lat for s in estaciones])
    lons = np.array([s.lon for s in estaciones])

    # No necesitamos unique() aquí si ya validamos Km estrictamente crecientes en cargar_estaciones
    # Pero lo mantenemos para robustez si esa validación fallara o se cambiara
    unique_kms, unique_indices = np.unique(kms, return_index=True)

    if len(unique_kms) < 2:
        logging.error("No hay suficientes puntos con Km únicos y crecientes para interpolar.")
        return []

    # Usar solo los puntos únicos y ordenados para la spline
    valid_kms = kms[unique_indices]
    valid_lats = lats[unique_indices]
    valid_lons = lons[unique_indices]

    try:
        spline_lat = CubicSpline(valid_kms, valid_lats)
        spline_lon = CubicSpline(valid_kms, valid_lons)
    except ValueError as e:
        logging.error(f"Error al crear splines cúbicas: {e}")
        return []

    km_inicio = valid_kms[0]
    km_fin = valid_kms[-1]
    if km_fin <= km_inicio:
        logging.error("Km final no es mayor que Km inicial después de filtrar y ordenar.")
        return []

    distancia_m = (km_fin - km_inicio) * 1000.0
    # Calcular el número de puntos basado en la distancia total y el intervalo
    num_puntos = max(2, int(np.ceil(distancia_m / intervalo_m)) + 1)

    # Generar los puntos de interpolación
    # Asegurarse de que el último punto sea exactamente km_fin
    kms_interp = np.linspace(km_inicio, km_fin, num=num_puntos, endpoint=True)

    # Asegurarse de que las latitudes y longitudes interpoladas estén dentro de rangos válidos
    lats_interp = spline_lat(kms_interp)
    lons_interp = spline_lon(kms_interp)

    puntos = [InterpolatedPoint(km=float(km_i), lat=float(lat_i), lon=float(lon_i))
              for km_i, lat_i, lon_i in zip(kms_interp, lats_interp, lons_interp)]

    # Filtrar puntos con coordenadas fuera de rango válidas después de interpolar
    puntos = [p for p in puntos if -90 <= p.lat <= 90 and -180 <= p.lon <= 180]

    logging.info(f"Interpolados {len(puntos)} puntos cada {intervalo_m} m entre Km {km_inicio:.3f} y {km_fin:.3f}")
    return puntos

# Configuración de reintento más robusta: espera exponencial + random jitter
# Espera base de 1s, multiplicador 2, max 60s. Jitter entre 0 y 5s.
@retry(
    stop=stop_after_attempt(7), # Intentar un poco más
    wait=wait_exponential(multiplier=1, min=5, max=60) + wait_random(0, 5) # Espera exponencial con jitter
)
def _fetch_elevation_batch_from_api(batch_coords: List[tuple[float, float]], timeout: int = REQUEST_TIMEOUT_SECONDS) -> List[float]:
    """Obtiene elevaciones para un lote de coordenadas desde la API de Open-Meteo con reintentos."""
    if not batch_coords:
        return []

    # Formatear coordenadas para la URL (5 decimales es suficiente precisión para la API)
    lats = [f"{c[0]:.5f}" for c in batch_coords]
    lons = [f"{c[1]:.5f}" for c in batch_coords]
    params = {'latitude': ','.join(lats), 'longitude': ','.join(lons)}

    try:
        response = requests.get(ELEVATION_API_URL, params=params, timeout=timeout)
        response.raise_for_status() # Esto lanzará una excepción para códigos de error (como 429 o 500)
        data = response.json()
        elevations = data.get("elevation", [])
        if not isinstance(elevations, list) or len(elevations) != len(batch_coords):
            # Si la API responde 200 pero los datos no son los esperados
             logging.error(f"API response format error for batch (first point: {lats[0]}, {lons[0]}). Data: {data}")
             # Considerar lanzar una excepción para reintentar el lote
             raise ValueError("Unexpected API response format") # Esto activará otro reintento de tenacity

        return [float(e) if isinstance(e, (int, float)) else DEFAULT_ELEVATION_ON_ERROR for e in elevations]
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for batch (first point: {lats[0]}, {lons[0]}): {e}")
        # La excepción será capturada por tenacity para reintentar
        raise
    except Exception as e:
        # Capturar otros errores inesperados y registrarlos
         logging.error(f"An unexpected error occurred during API fetch for batch (first point: {lats[0]}, {lons[0]}): {e}")
         # Si no es un RequestException (ej. ValueError por formato JSON), no reintentamos con tenacity
         # y asignamos valor por defecto a este lote.
         return [DEFAULT_ELEVATION_ON_ERROR] * len(batch_coords)


def obtener_elevaciones_paralelo(puntos: List[InterpolatedPoint],
                                  author: str = AUTHOR_ATTRIBUTION,
                                  progress_callback: Optional[callable] = None,
                                  max_workers: int = 4) -> List[InterpolatedPoint]:
    """Obtiene elevaciones usando caché y API por lotes en paralelo."""
    if not puntos:
        if progress_callback:
             progress_callback(1.0) # Marcar progreso completo si no hay puntos
        return []

    logging.info(f"Iniciando consulta de elevaciones para {len(puntos)} puntos...")
    puntos_con_elevacion = [None] * len(puntos)
    points_to_fetch_indices = [] # Lista de índices de puntos que NO están en caché

    # Paso 1: Cargar de caché
    num_total_puntos = len(puntos)
    for i, punto in enumerate(puntos):
        cached_elev = _load_elevation_from_cache(punto.lat, punto.lon)
        if cached_elev is not None:
            puntos_con_elevacion[i] = punto._replace(elevation=cached_elev)
        else:
            points_to_fetch_indices.append(i) # Guardar solo el índice

        if progress_callback:
            # Progreso inicial: porcentaje de puntos encontrados en caché
            # Asegurarse de que el progreso no supere 0.5 en esta fase
            progress_callback(min(0.5, (i + 1) / num_total_puntos * 0.5))

    num_cached = num_total_puntos - len(points_to_fetch_indices)
    logging.info(f"{num_cached} puntos encontrados en caché.")

    if not points_to_fetch_indices:
        logging.info("Todos los puntos encontrados en caché.")
        if progress_callback:
             progress_callback(1.0) # Marcar progreso completo
        # Asegurarse de que todos los puntos tengan elevación (deberían tenerla del caché)
        return [p if p is not None else puntos[i]._replace(elevation=DEFAULT_ELEVATION_ON_ERROR) for i, p in enumerate(puntos_con_elevacion)]


    # Paso 2: Preparar lotes para la API
    batches = []
    # Iterar sobre los índices de los puntos que necesitan ser obtenidos
    for i in range(0, len(points_to_fetch_indices), MAX_BATCH_SIZE):
        batch_indices_slice = points_to_fetch_indices[i:i + MAX_BATCH_SIZE]
        # Obtener las coordenadas correspondientes a estos índices
        batch_coords = [(puntos[idx].lat, puntos[idx].lon) for idx in batch_indices_slice]
        batches.append((batch_coords, batch_indices_slice)) # Guardar coords y sus índices originales

    logging.info(f"Se prepararon {len(batches)} lotes ({len(points_to_fetch_indices)} puntos) para consultar a la API.")

    # Paso 3: Consultar API en paralelo por lotes
    processed_count = 0
    # Asegurarse de que max_workers sea al menos 1 si hay lotes para procesar
    effective_workers = max(1, max_workers) if batches else 0

    if effective_workers > 0:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Mapear futuros a (coordenadas del lote, índices originales)
            futures = {executor.submit(_fetch_elevation_batch_from_api, coords): (coords, indices) for coords, indices in batches}
            for future in as_completed(futures):
                coords, indices = futures[future] # Recuperar las coords y los índices originales
                try:
                    elevations = future.result() # Obtener el resultado del lote
                    for point_idx_in_batch, original_idx in enumerate(indices):
                         # Asignar la elevación al punto correcto usando el índice original
                        elev = elevations[point_idx_in_batch] if point_idx_in_batch < len(elevations) else DEFAULT_ELEVATION_ON_ERROR
                        puntos_con_elevacion[original_idx] = puntos[original_idx]._replace(elevation=elev)
                        # Solo guardar en caché si la obtención fue exitosa (no valor por defecto) Y si la elevación es finita
                        if elev != DEFAULT_ELEVATION_ON_ERROR and np.isfinite(elev):
                             _save_elevation_to_cache(puntos[original_idx].lat, puntos[original_idx].lon, elev, author)

                except Exception as e:
                    # Esto capturará excepciones después de los reintentos de tenacity (si los hubo)
                    logging.error(f"Fallo final al procesar lote tras reintentos (primer punto: {coords[0] if coords else 'N/A'}, {coords[1] if coords and len(coords)>1 else 'N/A'}): {e}")
                    # Asignar valor por defecto a todos los puntos de este lote fallido
                    for original_idx in indices:
                        puntos_con_elevacion[original_idx] = puntos[original_idx]._replace(elevation=DEFAULT_ELEVATION_ON_ERROR)

                processed_count += len(indices)
                if progress_callback and len(points_to_fetch_indices) > 0:
                    # Progreso para la obtención de la API: 50% a 100%
                    # Asegurarse de que el progreso no supere 1.0
                    progress_callback(0.5 + min(0.5, processed_count / len(points_to_fetch_indices) * 0.5))


    # Paso 4: Llenar cualquier punto restante que haya fallado con el valor por defecto
    # Esto es una seguridad.
    for i, punto_final in enumerate(puntos_con_elevacion):
        if punto_final is None or punto_final.elevation is None or not np.isfinite(punto_final.elevation):
            puntos_con_elevacion[i] = puntos[i]._replace(elevation=DEFAULT_ELEVATION_ON_ERROR)

    logging.info("Consulta de elevaciones finalizada.")
    if progress_callback:
         progress_callback(1.0) # Marcar progreso completo al finalizar

    return puntos_con_elevacion


def calcular_pendiente_suavizada(kms: np.ndarray, elevs: np.ndarray, window_length: int) -> np.ndarray:
    """Calcula pendientes suavizadas (m/km) con filtro Savitzky-Golay."""
    # Asegurarse de que window_length sea impar y al menos 3
    # El valor se recibe del slider de Streamlit y ya se valida que sea impar y >= 3 allí.
    # Esta validación adicional es para seguridad si se llama la función directamente.
    if window_length % 2 == 0:
        window_length += 1 # Hacerlo impar
    if window_length < 3:
        window_length = 3 # Mínimo 3


    slope_m_per_km = np.full_like(elevs, np.nan, dtype=float) # Usar dtype float explícitamente
    kms_m = kms * 1000.0
    valid_indices = ~np.isnan(elevs) # Índices donde la elevación es un número válido

    # Necesitamos al menos 'window_length' puntos *válidos* para el filtro Savitzky-Golay
    num_valid = np.count_nonzero(valid_indices)
    if num_valid < window_length:
        logging.warning(f"No hay suficientes datos válidos ({num_valid}) para calcular pendiente con ventana {window_length}. Se necesitan al menos {window_length}. Retornando NaNs.")
        return slope_m_per_km # Retorna array de NaNs

    valid_kms_m = kms_m[valid_indices]
    valid_elevs = elevs[valid_indices]

    # Asegurarse de que la ventana efectiva no sea mayor que el número de puntos válidos
    effective_window = min(window_length, num_valid)
    # Asegurarse de que la ventana efectiva sea impar y al menos 3
    if effective_window % 2 == 0:
         effective_window -= 1 # Reducir a impar
    if effective_window < 3:
         logging.warning("Ventana efectiva para Savitzky-Golay es menor a 3. No se calculará la pendiente. Retornando NaNs.")
         return slope_m_per_km

    # El orden del polinomio debe ser menor que la ventana
    polyorder = min(2, effective_window - 1)
    if polyorder < 1:
         logging.warning("Orden del polinomio para Savitzky-Golay es menor a 1. No se calculará la pendiente. Retornando NaNs.")
         return slope_m_per_km


    # Asumiendo que los puntos interpolados están casi uniformemente espaciados en KM:
    # Calcular la diferencia promedio entre los KM válidos (en metros)
    delta_x_interp_valid = np.mean(np.diff(valid_kms_m))


    if np.isclose(delta_x_interp_valid, 0) or np.isnan(delta_x_interp_valid):
         logging.warning("Espaciado entre puntos interpolados válidos es cero o NaN. No se puede calcular la pendiente. Retornando NaNs.")
         return slope_m_per_km


    try:
        # Calcular la derivada (pendiente) usando el filtro Savitzky-Golay
        # delta = espaciado de la cuadrícula en la unidad del eje x (metros en valid_kms_m)
        gradient = savgol_filter(valid_elevs, window=effective_window, polyorder=polyorder, deriv=1, delta=delta_x_interp_valid)
        # Convertir pendiente de m/m a m/km (multiplicar por 1000)
        slope_m_per_km[valid_indices] = gradient * 1000.0 # Asignar pendientes solo a los índices válidos
    except Exception as e:
        logging.error(f"Error al calcular pendiente suavizada con Savitzky-Golay: {e}. Retornando NaNs.")
        return slope_m_per_km # Retornar NaNs en caso de error

    return slope_m_per_km

def graficar_html(puntos_visibles: List[InterpolatedPoint], # Cambiado para recibir solo puntos visibles
                  estaciones_tramo: List[Station],
                  titulo: str = "Perfil altimétrico",
                  slope_data_vis: Optional[np.ndarray] = None, # Cambiado para recibir pendientes solo de puntos visibles
                  theme: str = "light",
                  colors: str = "cyan,yellow", # Colores actualizados para mejor contraste en fondo negro
                  watermark: str = "LAL 2025") -> Optional[go.Figure]:
    """Genera gráfico interactivo con tema, colores y marca de agua, retornando el objeto Figure."""
    # Ahora la función recibe directamente los puntos visibles y las pendientes correspondientes
    if not puntos_visibles:
        logging.warning("No hay puntos visibles para graficar")
        return None

    kms = np.array([p.km for p in puntos_visibles])
    elevs = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_visibles], dtype=float)

    # Parsear colores, manejar errores o formato incorrecto
    try:
        elev_color, slope_color = colors.split(',')
        elev_color = elev_color.strip()
        slope_color = slope_color.strip()
        if not elev_color or not slope_color: # Si alguna cadena está vacía
             raise ValueError("Formato de colores incorrecto")
    except:
        logging.warning(f"Formato de colores '{colors}' incorrecto. Usando colores por defecto.")
        elev_color, slope_color = ("cyan", "yellow")


    has_slope_data = slope_data_vis is not None and isinstance(slope_data_vis, np.ndarray) and len(slope_data_vis) == len(kms)

    # Preparar texto de hover
    hover_texts = []
    for i, p in enumerate(puntos_visibles):
        km_text = f"<b>Km: {p.km:.3f}</b><br>"
        elev_text = f"Elev: {p.elevation:.1f} m" if p.elevation is not None and np.isfinite(p.elevation) else "Elev: N/A" # Usar isfinite
        slope_text = f"Pendiente: {slope_data_vis[i]:+.1f} m/km" if has_slope_data and i < len(slope_data_vis) and np.isfinite(slope_data_vis[i]) else "Pendiente: N/A" # Usar isfinite
        hover_texts.append(f"{km_text}{elev_text}<br>{slope_text}")


    fig = go.Figure()

    # Trazado de Elevación
    fig.add_trace(go.Scattergl(
        x=kms, y=elevs, mode='lines', name='Perfil', # Nombre cambiado a 'Perfil'
        line=dict(color=elev_color, width=2),
        hoverinfo='text', text=hover_texts, # Usamos el hover_texts preparado
        yaxis='y1'
    ))

    # Marcadores de Estación
    if estaciones_tramo:
        # Filtrar estaciones para mostrar solo las que están aproximadamente en el rango visible
        # Esto es una aproximación ya que el rango visible se define por Km
        min_km_vis = min(kms) if len(kms) > 0 else None
        max_km_vis = max(kms) if len(kms) > 0 else None

        estaciones_visibles = []
        if min_km_vis is not None and max_km_vis is not None:
            estaciones_visibles = [est for est in estaciones_tramo if est.km >= min_km_vis and est.km <= max_km_vis]

        if estaciones_visibles:
            station_kms = np.array([s.km for s in estaciones_visibles])
            # Intentar obtener elevación y pendiente de los puntos interpolados visibles más cercanos
            station_elevs = []
            station_slopes = [] # No se usa en hovertext de estación, pero se podría añadir
            station_hover_texts = []

            for est in estaciones_visibles:
                # Encontrar el punto interpolado visible con el KM más cercano a la estación
                closest_idx = np.argmin(np.abs(kms - est.km)) if len(kms) > 0 else None

                if closest_idx is not None:
                     elev = elevs[closest_idx] if not np.isnan(elevs[closest_idx]) else script2.DEFAULT_ELEVATION_ON_ERROR # Usar default de script2
                     # slope = slope_data_vis[closest_idx] if has_slope_data and closest_idx < len(slope_data_vis) and not np.isnan(slope_data_vis[closest_idx]) else "N/A" # No se usa en hovertext actual
                else:
                     # Si no hay puntos interpolados visibles, usar valor por defecto
                     elev = script2.DEFAULT_ELEVATION_ON_ERROR
                     # slope = "N/A"

                station_elevs.append(elev)
                # station_slopes.append(slope)
                station_hover_texts.append(f"<b>{est.nombre}</b><br>Km: {est.km:.3f}<br>Lat: {est.lat:.5f}<br>Lon: {est.lon:.5f}<br>Elev: {elev:.1f} m")


            fig.add_trace(go.Scatter(
                x=station_kms, y=station_elevs, # Usar los KMs originales de la estación
                mode='markers+text', text=[est.nombre for est in estaciones_visibles],
                textposition="top center",
                marker=dict(size=10, color='red', symbol='triangle-up', line=dict(width=1, color='white')), # Añadir borde blanco a marcadores
                name='Estaciones', # Nombre genérico para la leyenda de marcadores
                hoverinfo='text',
                hovertext=station_hover_texts,
                yaxis='y1'
            ))

    # Trazado de Pendiente
    if has_slope_data: # Usar slope_data_vis que ya está filtrada y calculada para los puntos visibles
        fig.add_trace(go.Scattergl(
            x=kms, y=slope_data_vis, mode='lines', name='Pendiente (m/km)',
            line=dict(color=slope_color, width=1.5, dash='dash'),
            yaxis='y2',
            hoverinfo='skip' # skip para evitar doble hover si ya está en el primer trace
        ))

    # Configuración del Layout
    # plotly_dark es buen template para fondo negro
    template = "plotly_dark" if theme.lower() == "dark" else "plotly" # Asegurar template oscuro para fondo negro
    annotations = [
        dict(
            text=watermark,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=60, color="rgba(255,255,255,0.1)"), # Color semi-transparente blanco para fondo negro
            textangle=-30,
            opacity=0.5
            # layer="below" # <-- ELIMINADA LA PROPIEDAD 'layer'
        )
    ] if watermark and watermark.lower() != "none" else []

    fig.update_layout(
        # Usar template completo para tema oscuro si se desea, o configurar individualmente
        template=template,
        # Configuración individual para asegurar fondo negro si template no es 'plotly_dark'
        # paper_bgcolor='black', # Comentado para dejar que el template lo defina
        # plot_bgcolor='black',  # Comentado para dejar que el template lo defina

        title=dict(text=titulo, x=0.5, xanchor='center', font=dict(size=18, color='white')),
        xaxis=dict(title='Kilómetro', color='white', showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=False, hoverformat='.3f'),
        yaxis=dict(title='Elevación (msnm)', tickfont=dict(color=elev_color), color='white', showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=False),
        yaxis2=dict(
            title="Pendiente (m/km)", tickfont=dict(color=slope_color), color='white',
            anchor="x", overlaying="y", side="right", showgrid=False # No mostrar grid para el eje secundario
        ),

        hovermode="x unified", # Modo hover unificado
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color='white')),
        margin=dict(l=60, r=60, t=90, b=50),

        # Añadir anotaciones (incluida la marca de agua si existe)
        annotations=annotations
    )

    # No guardar a archivo, solo retornar la figura
    return fig

# --- Funciones de Exportación (adaptadas para aceptar buffers o rutas) ---

def exportar_kml(puntos_con_elevacion: List[InterpolatedPoint], estaciones_tramo: List[Station], output: Union[str, io.BytesIO], author: str = AUTHOR_ATTRIBUTION):
    """Exporta estaciones a KML. Acepta ruta de archivo (str) o objeto archivo binario (io.BytesIO)."""
    kml = simplekml.Kml()
    for est in estaciones_tramo:
        # Intentar usar la elevación interpolada más cercana si está disponible
        closest_point = min(puntos_con_elevacion, key=lambda p: abs(p.km - est.km), default=None) if puntos_con_elevacion else None
        elev = closest_point.elevation if closest_point and closest_point.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
        # Usar las coordenadas originales de la estación para el punto KML
        pnt = kml.newpoint(name=est.nombre, coords=[(est.lon, est.lat, elev)], altitudemode='absolute') # Usar altitudemode='absolute'
        pnt.description = f"Km: {est.km:.3f}, Elevación: {elev:.1f} m\n{author}"

    try:
        # SimpleKML save() solo acepta filename (str).
        # Si el output es un BytesIO, guardamos a un tempfile y luego copiamos los bytes.
        if isinstance(output, str):
            kml.save(output)
            logging.info(f"KML guardado en: {output}")
        elif isinstance(output, io.BytesIO):
            # Guardar a un archivo temporal, leer y escribir al buffer
            # Usar with para asegurar que el tempfile se cierre y elimine
            with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp: # delete=False para que no se elimine al cerrar with
                temp_filename = tmp.name
            try:
                kml.save(temp_filename)
                with open(temp_filename, 'rb') as f:
                    output.write(f.read())
                # logging.info("KML guardado a BytesIO buffer.") # No loguear esto en bucle
            finally:
                # Asegurarse de eliminar el archivo temporal
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        else:
             raise TypeError("Output debe ser str (ruta de archivo) o io.BytesIO.")

    except Exception as e:
        logging.error(f"Error al guardar KML: {e}")
        raise # Propagar para manejar en Streamlit

def exportar_geojson(puntos_con_elevacion: List[InterpolatedPoint], estaciones_tramo: List[Station], output: Union[str, io.TextIOBase], author: str = AUTHOR_ATTRIBUTION):
    """Exporta puntos y estaciones a GeoJSON. Acepta ruta de archivo (str) o objeto archivo de texto (io.TextIOBase)."""
    features = []
    # Exportar puntos interpolados
    for p in puntos_con_elevacion:
        elev = p.elevation if p.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
        features.append(Feature(
            geometry=Point((p.lon, p.lat)), # GeoJSON es lon, lat
            properties={"km": p.km, "elevation": elev, "type": "interpolated", "author": author}
        ))
    # Exportar estaciones
    for est in estaciones_tramo:
        # Intentar usar la elevación interpolada más cercana si está disponible
        closest_point = min(puntos_con_elevacion, key=lambda p: abs(p.km - est.km), default=None) if puntos_con_elevacion else None
        elev = closest_point.elevation if closest_point and closest_point.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
        features.append(Feature(
            geometry=Point((est.lon, est.lat)), # Usar coords originales de la estación
            properties={"name": est.nombre, "km": est.km, "elevation": elev, "type": "station", "author": author}
        ))
    collection = FeatureCollection(features)

    try:
        # geojson.dump puede escribir a objetos archivo
        dump(collection, output, indent=2)
        # logging.info("GeoJSON data generated.") # No loguear en bucle
    except Exception as e:
        logging.error(f"Error al generar datos GeoJSON: {e}")
        raise # Propagar para manejar en Streamlit

def exportar_pdf(fig: go.Figure, output: Union[str, io.BytesIO]):
    """Exporta gráfico a PDF usando kaleido. Acepta ruta de archivo (str) o objeto archivo binario (io.BytesIO)."""
    try:
        # fig.write_image puede escribir a ruta de archivo o a BytesIO
        # https://plotly.com/python/static-image-export/
        fig.write_image(file=output, format="pdf", engine="kaleido")
        if isinstance(output, str):
            logging.info(f"PDF guardado en: {output}")
        # else: # No loguear si es a buffer
            # logging.info("PDF data generated to BytesIO buffer.")
    except Exception as e:
        logging.error(f"Error al guardar PDF: {e}. Asegúrate de tener kaleido instalado y operativo.")
        raise # Propagar para manejar en Streamlit


def exportar_csv(puntos_con_elevacion: List[InterpolatedPoint], slope_data: np.ndarray, output: Union[str, io.TextIOBase], author: str = AUTHOR_ATTRIBUTION):
    """Exporta datos interpolados a CSV. Acepta ruta de archivo (str) o objeto archivo de texto (io.TextIOBase)."""
    if not puntos_con_elevacion:
        logging.warning("No hay puntos para exportar a CSV.")
        return

    has_slope = slope_data is not None and isinstance(slope_data, np.ndarray) and len(slope_data) == len(puntos_con_elevacion)

    try:
        # csv.writer puede escribir a objetos archivo de texto
        writer = csv.writer(output) # output es el objeto archivo (o StringIO)
        writer.writerow([f"# {author}"])
        header = ["km", "latitude", "longitude", "elevation"]
        if has_slope:
             header.append("slope_m_per_km")
        writer.writerow(header)

        for i, p in enumerate(puntos_con_elevacion):
            elev = p.elevation if p.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
            row = [p.km, p.lat, p.lon, elev]
            if has_slope:
                 # Asegurarse de que el índice de slope_data sea válido antes de acceder
                 slope = slope_data[i] if i < len(slope_data) and not np.isnan(slope_data[i]) else '' # Usar cadena vacía para NaN en CSV
                 row.append(slope)
            writer.writerow(row)
        # logging.info("CSV data generated.") # No loguear en bucle

    except Exception as e:
        logging.error(f"Error al generar datos CSV: {e}")
        raise # Propagar para manejar en Streamlit

# --- Streamlit Cache para el caché en memoria ---
# La función initialize_elevation_cache y la asignación a script2._cache
# se manejan ahora en pantalla_loco.py.
