import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def media_movil(x, ventana):
    """Media móvil centrada de x con una ventana de `ventana` muestras.

    Devuelve un array del mismo largo que x. En los bordes promedia solo
    las muestras disponibles e ignora los NaN.
    """
    x = np.asarray(x, dtype=float)
    validos = ~np.isnan(x)
    kernel = np.ones(ventana)
    suma = np.convolve(np.where(validos, x, 0), kernel, mode='same')
    cuenta = np.convolve(validos.astype(float), kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        return suma / cuenta

CARPETA_DATOS = r'C:\Users\Milagritos\Desktop\TP_procesamiento\Datos\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0'
EPOCA = 30  # segundos (la grilla de los labels)


def cargar_sujeto(ID, carpeta=CARPETA_DATOS, epoca=EPOCA):
    """Carga los datos crudos de un sujeto y los lleva a épocas de `epoca` segundos.

    Devuelve un DataFrame con una fila por época etiquetada y columnas:
      tiempo      inicio de la época (s, relativo al inicio de la PSG)
      hr          frecuencia cardíaca media en la época (lpm)
      movimiento  desvío estándar de |aceleración| en la época (g)
      pasos       pasos del bloque de 10 min en el que cae la época
      etapa       etapa de sueño (0 vigilia, 1 N1, 2 N2, 3 N3, 4 N4, 5 REM)
    Las épocas sin dato de alguna señal quedan en NaN.
    """
    heart_rate = np.loadtxt(fr'{carpeta}\heart_rate\{ID}_heartrate.txt', delimiter=',')
    label = np.loadtxt(fr'{carpeta}\labels\{ID}_labeled_sleep.txt')
    motion = np.loadtxt(fr'{carpeta}\motion\{ID}_acceleration.txt')
    steps = np.loadtxt(fr'{carpeta}\steps\{ID}_steps.txt', delimiter=',')

    # Ordenar por tiempo (searchsorted / interp lo necesitan)
    heart_rate = heart_rate[np.argsort(heart_rate[:, 0])]
    motion = motion[np.argsort(motion[:, 0])]
    steps = steps[np.argsort(steps[:, 0])]

    # Sacar épocas sin etiqueta (-1)
    label = label[label[:, 1] != -1]
    epocas = label[:, 0]  # inicio de cada época válida

    # Heart rate: interpolar a 1 Hz dentro de cada época y promediar.
    # Fuera del rango registrado no se extrapola: queda NaN.
    t_1s = epocas[:, None] + np.arange(epoca)  # matriz (n_epocas, epoca)
    hr_1s = np.interp(t_1s, heart_rate[:, 0], heart_rate[:, 1],
                      left=np.nan, right=np.nan)
    with warnings.catch_warnings():  # épocas sin ningún dato -> NaN, sin warning
        warnings.simplefilter('ignore', RuntimeWarning)
        hr_epoca = np.nanmean(hr_1s, axis=1)

    # Motion: desvío estándar de la magnitud de la aceleración en cada época
    mag = np.linalg.norm(motion[:, 1:4], axis=1)
    ini = np.searchsorted(motion[:, 0], epocas)
    fin = np.searchsorted(motion[:, 0], epocas + epoca)
    mov_epoca = np.array([mag[i:f].std() if f > i else np.nan for i, f in zip(ini, fin)])

    # Steps: cada época toma el valor del bloque de 10 min en el que cae.
    # Si la época es anterior al primer bloque, NaN (evita que idx=-1 tome el último).
    idx_s = np.searchsorted(steps[:, 0], epocas, side='right') - 1
    steps_epoca = np.where(idx_s >= 0, steps[np.clip(idx_s, 0, None), 1], np.nan)

    return pd.DataFrame({'tiempo': epocas,
                         'hr': hr_epoca,
                         'movimiento': mov_epoca,
                         'pasos': steps_epoca,
                         'etapa': label[:, 1].astype(int)})


ID = 1455390
df = cargar_sujeto(ID)
print(df.head())
print(df.describe())
print('NaN por columna:\n', df.isna().sum())


# --- Media móvil de cada señal con las etapas de sueño (labels) ---
from matplotlib.patches import Patch

VENTANA = 1  # en épocas de 30 s (10 -> 5 min)
nombres_etapa = {0: 'Vigilia', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N3', 5: 'REM'}  # 4 = N4, se agrupa con N3
colores_etapa = {'Vigilia': 'tab:red', 'N1': 'tab:orange', 'N2': 'tab:cyan',
                 'N3': 'tab:blue', 'REM': 'tab:green'}
orden_hipno = ['N3', 'N2', 'N1', 'REM', 'Vigilia']  # de abajo hacia arriba

t_h = df['tiempo'] / 3600  # horas desde el inicio del registro
etapa_nombre = df['etapa'].map(nombres_etapa)


def graficar_con_etapas(columna, ylabel, titulo, ventana=VENTANA, log=False):
    """Señal por época + media móvil, con fondo coloreado por etapa e hipnograma abajo."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Fondo coloreado por etapa: un axvspan por cada tramo contiguo con la misma etapa
    cambios = np.flatnonzero(etapa_nombre.values[1:] != etapa_nombre.values[:-1]) + 1
    inicios = np.r_[0, cambios]
    fines = np.r_[cambios, len(df)]
    for i, f in zip(inicios, fines):
        e = etapa_nombre.iloc[i]
        t_fin = t_h.iloc[f - 1] + EPOCA / 3600
        ax1.axvspan(t_h.iloc[i], t_fin, color=colores_etapa[e], alpha=0.25, lw=0)

    duracion = ventana * EPOCA
    duracion_txt = f'{duracion // 60} min' if duracion >= 60 else f'{duracion} s'
    ax1.plot(t_h, df[columna], color='gray', alpha=0.4, lw=0.8, label='Valor por época')
    ax1.plot(t_h, media_movil(df[columna], ventana), color='black', lw=1.5,
             label=f'Media móvil ({duracion_txt})')
    ax1.set_ylabel(ylabel)
    if log:
        ax1.set_yscale('log')

    # Leyenda: líneas + parches de color de cada etapa
    handles, _ = ax1.get_legend_handles_labels()
    handles += [Patch(color=colores_etapa[e], alpha=0.4, label=e) for e in orden_hipno[::-1]]
    ax1.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)

    # Hipnograma
    y_hipno = etapa_nombre.map({e: k for k, e in enumerate(orden_hipno)})
    ax2.step(t_h, y_hipno, where='post', color='black', lw=1)
    ax2.set_yticks(range(len(orden_hipno)))
    ax2.set_yticklabels(orden_hipno)
    ax2.set_ylabel('Etapa')
    ax2.set_xlabel('Tiempo (h)')

    fig.suptitle(f'Sujeto {ID}: {titulo} y etapas de sueño')
    fig.tight_layout()
    return fig


graficar_con_etapas('hr', 'Frecuencia cardíaca (lpm)', 'frecuencia cardíaca')
graficar_con_etapas('movimiento', 'Desvío de |aceleración| (g)', 'movimiento', log=True)
plt.show()
