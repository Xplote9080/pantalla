import numpy as np
import plotly.graph_objects as go
import pandas as pd
from script2 import cargar_estaciones, interpolar_puntos, obtener_elevaciones_paralelo, calcular_pendiente_suavizada

# --- Configuración ---
ARCHIVO_CSV = "estaciones.csv"  # Tu archivo de entrada
KM_ACTUAL = 643.646              # Kilómetro actual donde está el tren
VENTANA_KM = 5                   # Cuántos km antes y después mostrar
SALIDA_HTML = "pantalla_loco.html"

# --- Cargar estaciones e interpolar ---
estaciones = cargar_estaciones(ARCHIVO_CSV)
puntos_interp = interpolar_puntos(estaciones, intervalo_m=100)
puntos_interp = obtener_elevaciones_paralelo(puntos_interp)

# --- Filtrar puntos cerca del km actual ---
kms = np.array([p.km for p in puntos_interp])
elevs = np.array([p.elevation for p in puntos_interp])
mask = (kms >= KM_ACTUAL - VENTANA_KM) & (kms <= KM_ACTUAL + VENTANA_KM)
kms_vis = kms[mask]
elevs_vis = elevs[mask]

# --- Calcular pendiente ---
pendientes = calcular_pendiente_suavizada(kms_vis, elevs_vis, window_length=5)

# --- Crear gráfico estilo cabina ---
fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=kms_vis,
    y=elevs_vis,
    mode='lines',
    line=dict(color='cyan', width=3),
    name='Perfil',
    hoverinfo='none'
))

# --- Línea del km actual ---
fig.add_shape(type='line', x0=KM_ACTUAL, x1=KM_ACTUAL,
              y0=min(elevs_vis), y1=max(elevs_vis),
              line=dict(color='red', width=3, dash='dot'))

# --- Texto de pendiente por tramo ---
for i in range(1, len(kms_vis)):
    mid_km = (kms_vis[i] + kms_vis[i-1]) / 2
    mid_elev = (elevs_vis[i] + elevs_vis[i-1]) / 2
    pendiente = pendientes[i]
    if np.isnan(pendiente):
        continue
    texto = f"{pendiente:+.1f}"  # signo + o -
    fig.add_annotation(x=mid_km, y=mid_elev+1,
                       text=texto,
                       showarrow=False,
                       font=dict(size=14, color='white'))

# --- Estilo oscuro tipo locomotora ---
fig.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white', size=18),
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis=dict(title='Kilómetro', color='white', showgrid=False),
    yaxis=dict(title='Elevación (m)', color='white', showgrid=False),
    title=dict(text=f"División DP - Km {KM_ACTUAL:.3f}", x=0.5, xanchor='center')
)

# --- Guardar HTML ---
fig.write_html(SALIDA_HTML, include_plotlyjs='cdn')
print(f"✅ Gráfico guardado como {SALIDA_HTML}")
