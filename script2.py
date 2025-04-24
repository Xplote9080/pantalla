# -*- coding: utf-8 -*-
"""
Script para generar perfiles altimétricos a partir de datos de estaciones para vías férreas.
© 2025 LAL - Todos los derechos reservados.

Funcionalidades:
- Carga estaciones desde un DataFrame (anteriormente CSV).
- Interpola puntos usando CubicSpline.
- Obtiene elevaciones con la API de Open-Meteo (con caché en CSV y llamadas por lotes paralelas).
- Calcula pendientes suavizadas (m/km) con filtro Savitzky-Golay.
- Genera gráficos HTML interactivos con Plotly (elevación + pendiente) con marca de agua.
- Exporta a PDF, KML, CSV y GeoJSON.
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
from plotly.offline import plot
import simplekml
from geojson import Point, Feature, FeatureCollection, dump
import logging
import os
import tempfile
import pandas as pd
from typing import List, Optional, NamedTuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, wait_random

# --- Configuración y Constantes ---
ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_CACHE_FILE = "elevations_cache.csv"
DEFAULT_INTERVAL_METERS = 200
REQUEST_TIMEOUT_SECONDS = 15
MAX_API_WORKERS = 4 # Se controla desde Streamlit
MAX_BATCH_SIZE = 50 # Reducido para mitigar error 429
DEFAULT_ELEVATION_ON_ERROR = 0.0
DEFAULT_SMOOTH_WINDOW = 5 # Ahora se controla desde Streamlit
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

# Usamos @st.cache_resource en Streamlit para cargar el caché una vez.
# Si usas este script fuera de Streamlit, la carga ocurrirá en el primer llamado.
# La lógica de guardado sigue siendo directa al archivo.
def load_cache_to_memory() -> Dict[tuple[float, float], float]:
    """Carga el caché de elevaciones en memoria desde el CSV."""
    global _cache
    if _cache is not None:
        return _cache # Si ya está cargado en memoria (ej. por Streamlit cache), retornarlo
    _cache = {}
    if not os.path.exists(ELEVATION_CACHE_FILE):
        logging.info("Archivo de caché no encontrado. Se creará uno nuevo.")
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
        logging.warning(f"Error al leer caché CSV: {e}")
    return _cache

def _load_elevation_from_cache(lat: float, lon: float) -> Optional[float]:
    """Carga elevación desde el caché en memoria."""
    cache = load_cache_to_memory() # Asegura que el caché esté cargado
    lat_r = round(lat, 5)
    lon_r = round(lon, 5)
    return cache.get((lat_r, lon_r))

def _save_elevation_to_cache(lat: float, lon: float, elevation: float, author: str = AUTHOR_ATTRIBUTION):
    """Guarda elevación en el archivo CSV y en memoria."""
    global _cache
    # Asegurarse de que el caché en memoria esté actualizado si se guarda
    lat_r = round(lat, 5)
    lon_r = round(lon, 5)

    # Bloquear accesos concurrentes si fuera necesario en un ambiente con múltiples hilos/procesos
    # Para Streamlit y este caso de uso, es menos crítico, pero es buena práctica
    # with cache_lock: # Si se implementara un RLock
    if (lat_r, lon_r) in _cache: # Evitar escribir duplicados si ya está en memoria
         return
    _cache[(lat_r, lon_r)] = elevation # Actualizar caché en memoria

    try:
        file_exists = os.path.exists(ELEVATION_CACHE_FILE)
        with open(ELEVATION_CACHE_FILE, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Escribir cabecera solo si el archivo es nuevo o está vacío
            if not file_exists or os.path.getsize(ELEVATION_CACHE_FILE) == 0:
                writer.writerow([f"# {author} - Caché de elevaciones para vías férreas"])
                writer.writerow(["latitude", "longitude", "elevation"])
            writer.writerow([lat_r, lon_r, elevation])
    except Exception as e:
        logging.warning(f"Error al guardar en caché CSV: {e}")

# --- Funciones principales ---
def cargar_estaciones(df: pd.DataFrame) -> List[Station]:
    """Carga estaciones desde un DataFrame de Pandas."""
    try:
        # Asegurarse de que las columnas tengan el formato esperado
        df.columns = [c.strip().capitalize() for c in df.columns]
        required_columns = {'Nombre', 'Km', 'Lat', 'Lon'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"El DataFrame debe contener las columnas: {required_columns}")

        estaciones = []
        # Usamos itertuples() que suele ser más rápido que iterrows()
        for row in df.itertuples(index=False):
            try:
                lat = float(row.Lat)
                lon = float(row.Lon)
                km = float(row.Km)
                nombre = str(row.Nombre).strip()

                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                     # Log solo la advertencia sin detener la ejecución por una fila mala
                    logging.warning(f"Coordenadas inválidas para '{nombre}': Lat={lat}, Lon={lon}. Fila saltada.")
                    continue # Saltar esta fila

                estaciones.append(Station(
                    nombre=nombre,
                    km=km,
                    lat=lat,
                    lon=lon
                ))
            except ValueError as ve:
                logging.warning(f"Error de valor al procesar fila: {row}. Error: {ve}. Fila saltada.")
                continue # Saltar esta fila si hay un error de conversión

        if len(estaciones) < 2:
            raise ValueError("Se requieren al menos 2 estaciones válidas después de la carga.")

        # Verificar que los kilómetros sean estrictamente crecientes después de cargar y filtrar
        if not all(estaciones[i].km < estaciones[i+1].km for i in range(len(estaciones)-1)):
             # Intenta ordenar y verificar si es posible, si no, lanza error
            try:
                estaciones_ordenadas = sorted(estaciones, key=lambda s: s.km)
                if not all(estaciones_ordenadas[i].km < estaciones_ordenadas[i+1].km for i in range(len(estaciones_ordenadas)-1)):
                     raise ValueError("Los kilómetros no son estrictamente crecientes incluso después de intentar ordenar.")
                estaciones = estaciones_ordenadas # Usar la lista ordenada si fue posible
            except Exception:
                 raise ValueError("Los kilómetros deben ser estrictamente crecientes.")


        logging.info(f"Cargadas {len(estaciones)} estaciones válidas desde el DataFrame")
        return estaciones
    except Exception as e:
        logging.error(f"Error al cargar estaciones desde DataFrame: {e}")
        raise # Propagar la excepción para que Streamlit la muestre

def interpolar_puntos(estaciones: List[Station], intervalo_m: int = DEFAULT_INTERVAL_METERS) -> List[InterpolatedPoint]:
    """Interpola puntos geográficos."""
    if len(estaciones) < 2:
        logging.warning("Mínimo 2 estaciones requeridas para interpolación")
        return []

    # Asegurarse de que las estaciones estén ordenadas por Km
    estaciones_ordenadas = sorted(estaciones, key=lambda s: s.km)

    kms = np.array([s.km for s in estaciones_ordenadas])
    lats = np.array([s.lat for s in estaciones_ordenadas])
    lons = np.array([s.lon for s in estaciones_ordenadas])

    # Usar np.unique para manejar posibles duplicados de Km, aunque ya validamos
    # estrictamente crecientes en cargar_estaciones. Esto añade robustez.
    unique_kms, unique_indices = np.unique(kms, return_index=True)

    # Si hay menos de 2 puntos únicos después de la validación y unique, algo salió mal
    if len(unique_kms) < 2:
        logging.error("No hay suficientes puntos válidos con Km únicos y crecientes para interpolar.")
        return []

    valid_lats = lats[unique_indices]
    valid_lons = lons[unique_indices]

    try:
        # Usar los puntos únicos y ordenados para la spline
        spline_lat = CubicSpline(unique_kms, valid_lats)
        spline_lon = CubicSpline(unique_kms, valid_lons)
    except ValueError as e:
        logging.error(f"Error al crear splines cúbicas: {e}")
        return []

    km_inicio = unique_kms[0]
    km_fin = unique_kms[-1]
    if km_fin <= km_inicio:
        logging.error("Km final no es mayor que Km inicial después de filtrar y ordenar.")
        return []

    distancia_m = (km_fin - km_inicio) * 1000.0
    # Calcular el número de puntos basado en la distancia total y el intervalo
    num_puntos = max(2, int(np.ceil(distancia_m / intervalo_m)) + 1)

    # Generar los puntos de interpolación
    kms_interp = np.linspace(km_inicio, km_fin, num_puntos)

    # Asegurarse de incluir exactamente el punto final si no está ya en linspace (raro con float, pero seguridad)
    if abs(kms_interp[-1] - km_fin) > 1e-9 * abs(km_fin): # Tolerancia para comparación float
         kms_interp = np.append(kms_interp, km_fin)
         kms_interp = np.unique(kms_interp) # Asegurar unicidad y orden tras append

    lats_interp = spline_lat(kms_interp)
    lons_interp = spline_lon(kms_interp)

    puntos = [InterpolatedPoint(km=float(km_i), lat=float(lat_i), lon=float(lon_i))
              for km_i, lat_i, lon_i in zip(kms_interp, lats_interp, lons_interp)]

    # Filtrar puntos con coordenadas fuera de rango válidas después de interpolar
    puntos = [p for p in puntos if -90 <= p.lat <= 90 and -180 <= p.lon <= 180]

    logging.info(f"Interpolados {len(puntos)} puntos cada {intervalo_m} m entre Km {km_inicio:.3f} y {km_fin:.3f}")
    return puntos

# Configuración de reintento más robusta: espera exponencial + random jitter
# Espera base de 1s, multiplicador 2, max 60s. Jitter entre 0 y 2s.
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
             # Podrías considerar lanzar una excepción para reintentar, o asignar valores por defecto
             # raise ValueError("Unexpected API response format")
             # Por ahora, asignamos por defecto si el formato es incorrecto
             return [DEFAULT_ELEVATION_ON_ERROR] * len(batch_coords)

        return [float(e) if isinstance(e, (int, float)) else DEFAULT_ELEVATION_ON_ERROR for e in elevations]
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for batch (first point: {lats[0]}, {lons[0]}): {e}")
        # La excepción será capturada por tenacity para reintentar
        raise
    except Exception as e:
        # Capturar otros errores inesperados y registrarlos
         logging.error(f"An unexpected error occurred during API fetch for batch (first point: {lats[0]}, {lons[0]}): {e}")
         # No lanzamos para reintentar si no es un RequestException, sino que asignamos valor por defecto
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
            progress_callback(i / num_total_puntos * 0.5) # 0% a 50% para caché

    num_cached = num_total_puntos - len(points_to_fetch_indices)
    logging.info(f"{num_cached} puntos encontrados en caché.")

    if not points_to_fetch_indices:
        logging.info("Todos los puntos encontrados en caché.")
        if progress_callback:
             progress_callback(1.0) # Marcar progreso completo
        return puntos_con_elevacion

    # Paso 2: Preparar lotes para la API
    batches = []
    # Iterar sobre los índices de los puntos que necesitan ser obtenidos
    for i in range(0, len(points_to_fetch_indices), MAX_BATCH_SIZE):
        batch_indices_slice = points_to_fetch_indices[i:i + MAX_BATCH_SIZE]
        # Obtener las coordenadas correspondientes a estos índices
        batch_coords = [(puntos[idx].lat, puntos[idx].lon) for idx in batch_indices_slice]
        batches.append((batch_coords, batch_indices_slice)) # Guardar coords y sus índices originales

    logging.info(f"Se prepararon {len(batches)} lotes para consultar a la API.")

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
                        if elev != DEFAULT_ELEVATION_ON_ERROR: # Solo guardar en caché si la obtención fue exitosa (no valor por defecto)
                             _save_elevation_to_cache(puntos[original_idx].lat, puntos[original_idx].lon, elev, author)

                except Exception as e:
                    # Esto capturará excepciones después de los reintentos de tenacity
                    logging.error(f"Fallo final al procesar lote (primer punto: {coords[0] if coords else 'N/A'}, {coords[1] if coords and len(coords)>1 else 'N/A'}): {e}")
                    # Asignar valor por defecto a todos los puntos de este lote fallido
                    for original_idx in indices:
                        puntos_con_elevacion[original_idx] = puntos[original_idx]._replace(elevation=DEFAULT_ELEVATION_ON_ERROR)

                processed_count += len(indices)
                if progress_callback and len(points_to_fetch_indices) > 0:
                    # Progreso para la obtención de la API: 50% a 100%
                    progress_callback(0.5 + (processed_count / len(points_to_fetch_indices)) * 0.5)


    # Paso 4: Llenar cualquier punto restante que haya fallado con el valor por defecto
    # Esto es una seguridad, ya que los puntos fallidos en el try/except ya deberían tener valor por defecto.
    for i, punto_final in enumerate(puntos_con_elevacion):
        if punto_final is None or punto_final.elevation is None:
            puntos_con_elevacion[i] = puntos[i]._replace(elevation=DEFAULT_ELEVATION_ON_ERROR)

    logging.info("Consulta de elevaciones finalizada.")
    if progress_callback:
         progress_callback(1.0) # Marcar progreso completo al finalizar

    return puntos_con_elevacion


def calcular_pendiente_suavizada(kms: np.ndarray, elevs: np.ndarray, window_length: int = DEFAULT_SMOOTH_WINDOW) -> np.ndarray:
    """Calcula pendientes suavizadas (m/km) con filtro Savitzky-Golay."""
    # Asegurarse de que window_length sea impar y al menos 3
    if window_length % 2 == 0:
        window_length += 1
    if window_length < 3:
        window_length = 3

    slope_m_per_km = np.full_like(elevs, np.nan, dtype=float) # Usar dtype float explícitamente
    kms_m = kms * 1000.0
    valid_indices = ~np.isnan(elevs)

    # Necesitamos al menos 'window_length' puntos válidos para el filtro Savitzky-Golay
    num_valid = np.count_nonzero(valid_indices)
    if num_valid < window_length:
        logging.warning(f"No hay suficientes datos válidos ({num_valid}) para calcular pendiente con ventana {window_length}. Se necesitan al menos {window_length}.")
        return slope_m_per_km # Retorna array de NaNs

    valid_kms_m = kms_m[valid_indices]
    valid_elevs = elevs[valid_indices]

    # Asegurarse de que la ventana no sea mayor que el número de puntos válidos
    effective_window = min(window_length, num_valid)
    # Asegurarse de que la ventana efectiva sea impar y al menos 3
    if effective_window % 2 == 0:
         effective_window -= 1 # Reducir a impar
    if effective_window < 3:
         # Esto no debería ocurrir si num_valid >= window_length >= 3, pero seguridad
         logging.warning("Ventana efectiva para Savitzky-Golay es menor a 3. No se calculará la pendiente.")
         return slope_m_per_km

    # El orden del polinomio debe ser menor que la ventana
    polyorder = min(2, effective_window - 1)
    if polyorder < 1: # Se necesita al menos orden 1 para calcular la derivada
         logging.warning("Orden del polinomio para Savitzky-Golay es menor a 1. No se calculará la pendiente.")
         return slope_m_per_km


    # Calcular el delta x promedio para el filtro
    # Si los puntos no están perfectamente espaciados, usar np.gradient es más robusto que Savitzky-Golay con delta fijo.
    # Sin embargo, Savitzky-Golay con delta fijo es más común para suavizado de derivada.
    # Mantenemos Savitzky-Golay pero calculamos un delta representativo.
    # Un enfoque alternativo y más robusto para datos no uniformes sería usar np.gradient.
    # delta_x = np.mean(np.diff(valid_kms_m)) # Usar delta promedio

    # Para Savitzky-Golay, el delta es el espaciado de la cuadrícula subyacente.
    # Como interpolamos a intervalos fijos, el delta debería ser constante.
    # Si los puntos de entrada tienen un espaciado ligeramente variable, el delta puede ser un promedio o el más común.
    # Asumiendo que los puntos interpolados están casi uniformemente espaciados:
    delta_x_interp = np.mean(np.diff(kms_m)) # Delta x basado en los kms interpolados en metros

    if np.isclose(delta_x_interp, 0):
         logging.warning("Espaciado entre puntos interpolados es cero. No se puede calcular la pendiente.")
         return slope_m_per_km


    try:
        # Calcular la derivada (pendiente) usando el filtro Savitzky-Golay
        # delta = espaciado de la cuadrícula en la unidad del eje x (metros en valid_kms_m)
        gradient = savgol_filter(valid_elevs, window=effective_window, polyorder=polyorder, deriv=1, delta=delta_x_interp)
        # Convertir pendiente de m/m a m/km (multiplicar por 1000)
        slope_m_per_km[valid_indices] = gradient * 1000.0
    except Exception as e:
        logging.error(f"Error al calcular pendiente suavizada con Savitzky-Golay: {e}")

    return slope_m_per_km

def graficar_html(puntos_con_elevacion: List[InterpolatedPoint],
                  estaciones_tramo: List[Station],
                  archivo_html: str,
                  titulo: str = "Perfil altimétrico",
                  slope_data: Optional[np.ndarray] = None,
                  theme: str = "light",
                  colors: str = "blue,orange",
                  watermark: str = "LAL 2025") -> Optional[go.Figure]:
    """Genera gráfico interactivo con tema, colores y marca de agua."""
    if not puntos_con_elevacion:
        logging.warning("No hay puntos para graficar")
        return None

    kms = np.array([p.km for p in puntos_con_elevacion])
    # Convertir None a NaN para permitir funciones numpy/pandas
    elevs = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elevacion], dtype=float)
    elev_color, slope_color = colors.split(',') if ',' in colors and len(colors.split(',')) == 2 else ("blue", "orange")

    # Asegurarse de que los datos de pendiente coincidan en longitud y sean np.ndarray
    has_slope_data = slope_data is not None and isinstance(slope_data, np.ndarray) and len(slope_data) == len(kms)

    # Preparar texto de hover
    hover_texts = []
    for i, p in enumerate(puntos_con_elevacion):
        km_text = f"<b>Km: {p.km:.3f}</b><br>"
        elev_text = f"Elev: {p.elevation:.1f} m" if p.elevation is not None and not np.isnan(p.elevation) else "Elev: N/A"
        slope_text = f"Pendiente: {slope_data[i]:+.1f} m/km" if has_slope_data and not np.isnan(slope_data[i]) else "Pendiente: N/A"
        hover_texts.append(f"{km_text}{elev_text}<br>{slope_text}")


    fig = go.Figure()

    # Trazado de Elevación
    fig.add_trace(go.Scattergl(
        x=kms, y=elevs, mode='lines', name='Elevación',
        line=dict(color=elev_color, width=2),
        hoverinfo='text', text=hover_texts, # Usamos el hover_texts preparado
        yaxis='y1'
    ))

    # Marcadores de Estación
    if estaciones_tramo:
        station_kms = np.array([s.km for s in estaciones_tramo])
        station_lats = np.array([s.lat for s in estaciones_tramo]) # Usar coords originales de estación
        station_lons = np.array([s.lon for s in estaciones_tramo])

        station_elevs = []
        station_slopes = []
        station_hover_texts = []

        # Intentar obtener elevación y pendiente de los puntos interpolados más cercanos
        for est in estaciones_tramo:
            # Encontrar el punto interpolado con el KM más cercano a la estación
            closest_idx = np.argmin(np.abs(kms - est.km)) if len(kms) > 0 else None

            if closest_idx is not None:
                 elev = elevs[closest_idx] if not np.isnan(elevs[closest_idx]) else DEFAULT_ELEVATION_ON_ERROR
                 slope = slope_data[closest_idx] if has_slope_data and not np.isnan(slope_data[closest_idx]) else "N/A"
            else:
                 # Si no hay puntos interpolados, usar valor por defecto
                 elev = DEFAULT_ELEVATION_ON_ERROR
                 slope = "N/A"

            station_elevs.append(elev)
            station_slopes.append(slope)
            station_hover_texts.append(f"<b>{est.nombre}</b><br>Km: {est.km:.3f}<br>Lat: {est.lat:.5f}<br>Lon: {est.lon:.5f}<br>Elev: {elev:.1f} m<br>Pendiente: {slope} m/km")


        fig.add_trace(go.Scatter(
            x=station_kms, y=station_elevs, # Usar los KMs originales de la estación
            mode='markers+text', text=[est.nombre for est in estaciones_tramo],
            textposition="top center",
            marker=dict(size=10, color='red', symbol='triangle-up'),
            name='Estaciones', # Nombre genérico para la leyenda de marcadores
            hoverinfo='text',
            hovertext=station_hover_texts,
            yaxis='y1'
        ))

    # Trazado de Pendiente
    if has_slope_data:
        fig.add_trace(go.Scattergl(
            x=kms, y=slope_data, mode='lines', name='Pendiente (m/km)',
            line=dict(color=slope_color, width=1.5, dash='dash'),
            yaxis='y2',
             # Puedes añadir hoverinfo='text' aquí si quieres un hover específico para la pendiente
            hoverinfo='skip' # skip para evitar doble hover si ya está en el primer trace
        ))

    # Configuración del Layout
    template = "plotly" if theme == "light" else "plotly_dark"
    annotations = [
        dict(
            text=watermark,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=80, color="rgba(150, 150, 150, 0.3)"),
            textangle=-45,
            opacity=0.3,
            layer="below" # Asegura que la marca de agua esté detrás de los datos
        )
    ] if watermark and watermark.lower() != "none" else []

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor='center'),
        xaxis_title="Kilómetro",
        yaxis=dict(title="Elevación (msnm)", tickfont=dict(color=elev_color), showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
        yaxis2=dict(
            title="Pendiente (m/km)", tickfont=dict(color=slope_color),
            anchor="x", overlaying="y", side="right", showgrid=False # No mostrar grid para el eje secundario
        ),
        xaxis=dict(hoverformat='.3f', showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
        hovermode="x unified", # Modo hover unificado para mostrar datos de ambos ejes
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=90, b=50),
        template=template,
        annotations=annotations
    )

    try:
        # Para Streamlit, no guardamos en archivo HTML, sino que retornamos la figura
        # plot(fig, filename=archivo_html, auto_open=False, include_plotlyjs='cdn')
        # logging.info(f"Gráfico (no guardado en HTML para Streamlit)")
        pass # No hacemos nada aquí ya que Streamlit renderizará la figura
    except Exception as e:
        logging.error(f"Error (ignorado) al generar figura Plotly: {e}") # Log como error ignorado si no es crítico

    return fig

# --- Funciones de Exportación (mantidas, pero podrían adaptarse si se usan) ---

def exportar_kml(puntos_con_elevacion: List[InterpolatedPoint], estaciones_tramo: List[Station], archivo_kml: str, author: str = AUTHOR_ATTRIBUTION):
    """Exporta estaciones a KML."""
    kml = simplekml.Kml()
    for est in estaciones_tramo:
        # Intentar usar la elevación interpolada más cercana si está disponible
        closest_point = min(puntos_con_elevacion, key=lambda p: abs(p.km - est.km), default=None) if puntos_con_elevacion else None
        elev = closest_point.elevation if closest_point and closest_point.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
        # Usar las coordenadas originales de la estación para el punto KML
        pnt = kml.newpoint(name=est.nombre, coords=[(est.lon, est.lat, elev)], altitudemode='absolute') # Usar altitudemode='absolute'
        pnt.description = f"Km: {est.km:.3f}, Elevación: {elev:.1f} m\n{author}"
    try:
        kml.save(archivo_kml)
        logging.info(f"KML guardado en: {archivo_kml}")
    except Exception as e:
        logging.error(f"Error al guardar KML: {e}")

def exportar_geojson(puntos_con_elevacion: List[InterpolatedPoint], estaciones_tramo: List[Station], archivo_geojson: str, author: str = AUTHOR_ATTRIBUTION):
    """Exporta puntos y estaciones a GeoJSON."""
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
        with open(archivo_geojson, 'w', encoding='utf-8') as f:
            dump(collection, f, indent=2)
        logging.info(f"GeoJSON guardado en: {archivo_geojson}")
    except Exception as e:
        logging.error(f"Error al guardar GeoJSON: {e}")

def exportar_pdf(fig: go.Figure, archivo_pdf: str):
    """Exporta gráfico a PDF usando kaleido."""
    # Esta función asume que 'fig' es un objeto Plotly Figure.
    # Kaleido necesita ser instalado (está en requirements.txt).
    try:
        fig.write_image(archivo_pdf, engine="kaleido")
        logging.info(f"PDF guardado en: {archivo_pdf}")
    except Exception as e:
        logging.error(f"Error al guardar PDF: {e}. Asegúrate de tener kaleido instalado y operativo.")

def exportar_csv(puntos_con_elevacion: List[InterpolatedPoint], slope_data: np.ndarray, archivo_csv: str, author: str = AUTHOR_ATTRIBUTION):
    """Exporta datos interpolados a CSV."""
    if not puntos_con_elevacion:
        logging.warning("No hay puntos para exportar a CSV.")
        return

    has_slope = slope_data is not None and isinstance(slope_data, np.ndarray) and len(slope_data) == len(puntos_con_elevacion)

    try:
        with open(archivo_csv, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"# {author}"])
            header = ["km", "latitude", "longitude", "elevation"]
            if has_slope:
                 header.append("slope_m_per_km")
            writer.writerow(header)

            for i, p in enumerate(puntos_con_elevacion):
                elev = p.elevation if p.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
                row = [p.km, p.lat, p.lon, elev]
                if has_slope:
                     slope = slope_data[i] if not np.isnan(slope_data[i]) else '' # Usar cadena vacía para NaN en CSV
                     row.append(slope)
                writer.writerow(row)
        logging.info(f"CSV guardado en: {archivo_csv}")
    except Exception as e:
        logging.error(f"Error al guardar CSV: {e}")

# --- Streamlit Cache para el caché en memoria ---
# Esto DEBE estar en el archivo principal de Streamlit (pantalla_loco.py),
# pero lo pongo aquí comentado como recordatorio de cómo usarlo si se quisiera
# inicializar el caché en memoria al inicio de la app de forma eficiente.
# @st.cache_resource
# def initialize_elevation_cache():
#     """Inicializa y carga el caché de elevaciones usando la función del script2."""
#     logging.info("Inicializando caché de elevaciones para Streamlit...")
#     return load_cache_to_memory()

# # En tu pantalla_loco.py, llamarías:
# # elevation_cache = initialize_elevation_cache()
# # script2._cache = elevation_cache # Asignar el caché cargado a la variable global del script2
# # Esto asegura que el caché se carga una vez por despliegue/reinicio de Streamlit.
# # La función _save_elevation_to_cache actualizará esta misma variable en memoria.

# Note: La implementación actual de load_cache_to_memory en script2.py
# es suficientemente segura porque verifica si _cache ya tiene datos.
# La @st.cache_resource en pantalla_loco.py llamando a esta función
# sería la forma más "correcta" de hacerlo en el contexto de Streamlit
# para garantizar que solo se cargue una vez por sesión/despliegue.
# He mantenido la lógica de _cache = None y la verificación dentro de load_cache_to_memory
# para que el script sea más autocontenido, pero la combinación con
# @st.cache_resource en pantalla_loco.py es lo ideal en ese entorno.