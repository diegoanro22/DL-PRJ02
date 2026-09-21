# Proyecto 2 — Competencia de Agentes en Space Invaders

**CC3092 Deep Learning y Sistemas Inteligentes**

Agente de Reinforcement Learning (PPO) sobre `ALE/SpaceInvaders-v5`.

**Resultado final: 2,866 puntos de media** (30 episodios, política determinista,
3 vidas, recompensa sin recortar), con máximo de 3,035.

| Referencia | Puntaje |
|---|---|
| Agente aleatorio | 166 |
| Agente de regla simple | 245 |
| QRDQN (este proyecto, 25M pasos) | 1,088 |
| Humano promedio | ~1,670 |
| DQN (Mnih et al., 2015) | ~1,976 |
| Double DQN | ~2,525 |
| **PPO (este proyecto, 100M pasos)** | **2,866** |

## Requisitos

Python **3.12** — PyTorch y Stable-Baselines3 no publican wheels para 3.14.
GPU NVIDIA con soporte CUDA 12.8 (probado en RTX 5070 Laptop, Blackwell sm_120).

## Instalación

```bash
uv venv --python 3.12 .venv

# torch va en un paso aparte, desde el indice de PyTorch.
# Mezclarlo con el resto rompe la resolucion de numpy.
uv pip install --python .venv/bin/python torch==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python .venv/bin/python -r requirements.txt
```

Verificar que la GPU ejecuta convoluciones antes de entrenar:

```bash
.venv/bin/python -c "
import torch, torch.nn as nn
print(torch.__version__, torch.cuda.get_device_capability(0))
print(nn.Conv2d(4,32,8,4).cuda()(torch.randn(8,4,84,84,device='cuda')).shape)"
```

## Evaluar el agente entrenado

Un solo comando. Carga los pesos y la configuración de wrappers, corre los
episodios con política determinista, imprime la tabla de puntajes y graba el video:

```bash
.venv/bin/python src/evaluar.py --agente artifacts/agente_final.zip --episodios 5 --video
```

Los pesos están en `artifacts/agente_final.zip` y **la configuración completa de
wrappers en `artifacts/agente_final.config.yaml`**, que `evaluar.py` lee
automáticamente. Stable-Baselines3 no guarda el stack de wrappers dentro del
`.zip`, así que ese YAML es imprescindible para reconstruir el entorno correcto.

## Reproducir el entrenamiento

```bash
# PPO, 16 entornos paralelos, 100M pasos (~8 h en RTX 5070 Laptop)
.venv/bin/python src/entrenar_sb3.py --config configs/ppo.yaml --run-id ppo_01

# QRDQN, 25M pasos
.venv/bin/python src/entrenar_sb3.py --config configs/qrdqn.yaml --run-id qrdqn_01
```

Progreso en vivo: `./estado.sh`

> **Sobre `--resume`.** La bandera existe y se usó para IT-04b (continuar un run ya
> terminado con hiperparámetros nuevos), pero **no es apta para reanudar un
> entrenamiento interrumpido**. Stable-Baselines3 interpreta `total_timesteps` como
> pasos *adicionales* y no como objetivo final, de modo que un run caído en 40M
> reanudado con un config de 100M entrenaría hasta 140M. Además, sobre algoritmos
> off-policy falla al reconstruir `train_freq` y no regenera el calendario de
> exploración. Estas limitaciones se documentan como trabajo pendiente en el informe;
> mientras tanto, para reanudar conviene ajustar `total_timesteps` al número de pasos
> que realmente faltan.

## Selección del checkpoint final

```bash
# Etapa 1: filtro sobre todos los checkpoints
.venv/bin/python src/embudo.py --run-id ppo_01 --episodios 10 --semilla-base 9000

# Etapa 2: finalistas con más episodios
.venv/bin/python src/embudo.py --run-id ppo_01 --episodios 30 --semilla-base 7000 \
  --solo artifacts/checkpoints/ppo_01/paso_95000000_steps.zip ...
```

Reporta media, desviación, mediana, máximo, mínimo y `E[máx de 5]` por bootstrap,
porque el enunciado se contradice sobre cuál métrica define el ranking.

## Verificación crítica

El entrenamiento usa un `VecEnv` de Stable-Baselines3 y la evaluación un entorno
Gymnasium normal (para poder reutilizar `ejecutar_episodio` y `RecordVideo` del
Laboratorio 5). **Las dos rutas deben producir observaciones idénticas** o el
agente jugaría con una entrada distinta a la que vio entrenando, sin lanzar
ningún error:

```bash
.venv/bin/python src/verificar_equivalencia.py
# RESULTADO: las dos rutas son IDÉNTICAS.
```

## Estructura

```
src/
  ale_interaccion.py        # portado del Laboratorio 5 (+ parámetro fabrica_entorno)
  entorno.py                # fábricas de entorno: ruta SB3 y ruta Gymnasium
  entrenar_sb3.py           # entrenamiento, reanudable
  embudo.py                 # selección de checkpoint en dos etapas
  evaluar.py                # evaluación final + video, un solo comando
  verificar_equivalencia.py # prueba de equivalencia entre las dos rutas
configs/                    # hiperparámetros por experimento
artifacts/
  agente_final.zip          # pesos del agente de competencia
  agente_final.config.yaml  # configuración de wrappers (imprescindible)
  evaluaciones/             # resultados crudos y hallazgos
videos/agente_final.mp4     # episodio completo del agente
resultados_iteraciones.csv  # historial de experimentos
```

## Detalles de implementación que importan

- `ALE/SpaceInvaders-v5` trae `frameskip=4` de fábrica y los wrappers de Atari
  aplican otro salto de 4. Se pasa `frameskip=1` al entorno para no acumular un
  salto efectivo de 16, que impide aprender sin dar ningún error.
- Sticky actions (`repeat_action_probability=0.25`) activadas en entrenamiento y
  evaluación, porque es el default de v5 y es lo que habrá en la evaluación real.
- Entrenamiento con vidas episódicas y recompensa recortada; evaluación con las
  3 vidas completas y recompensa cruda.
- El apilado de frames usa `padding_type="zero"` para replicar `VecFrameStack`
  de SB3, cuyo default difiere del de Gymnasium.
