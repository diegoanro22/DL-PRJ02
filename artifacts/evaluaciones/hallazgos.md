# Hallazgos para el informe

## H1 — Distribución bimodal del agente final (sección 2.4)

30 episodios deterministas del checkpoint de 95M:

- Grupo 1: 23 episodios (77%) entre 2765 y 2840, moda en 2830-2835.
- Grupo 2: 7 episodios (23%) entre 2995 y 3035.
- Ningún episodio entre 2840 y 2995.

La mediana de los grupos difiere en 185 puntos, pero la correspondencia exacta se ve
valor por valor: cada puntaje del grupo alto es uno del grupo bajo **más 200**, el
valor de la nave nodriza (UFO) en Space Invaders. Cada valor del grupo 2 equivale a uno del grupo 1
más 200: 2835+200=3035, 2830+200=3030, 2795+200=2995.

**Interpretación:** el agente ejecuta esencialmente la misma política en cada
partida y llega de forma consistente a ~2835 puntos; la variación observada no es
"jugar mejor o peor", sino acertarle o no a la nave nodriza una vez.

Cada formación limpia vale 630 puntos. 2835 / 630 = 4.5, es decir que limpia
cuatro oleadas completas y muere a mitad de la quinta, casi siempre en el mismo
punto.

**Vía de mejora identificada:** priorizar la nave nodriza valdría +200 por partida,
más que cualquier ajuste de hiperparámetros en este punto de la convergencia.

## H2 — La política estocástica no mejora el máximo (sección 2.4)

Hipótesis: al muestrear en vez de tomar argmax, sube la varianza y con ella el
máximo de 5 episodios (relevante si el ranking usa el máximo).

Medición sobre las mismas 30 semillas:

| Política | media | std | máx | E[máx de 5] |
|---|---|---|---|---|
| Determinista | 2866 | 87 | 3035 | 2974 |
| Estocástica | 2849 | 103 | 3035 | 2966 |

**Refutada.** La varianza sube (87 -> 103) pero el máximo no se mueve y la media
baja. Se usa política determinista.

## H3 — best_model de SB3 no era el mejor checkpoint (sección 2.3)

`EvalCallback` guarda `best_model` según la recompensa **media** de solo 5
episodios. En el embudo de 22 checkpoints x 10 episodios, `best_model.zip` quedó
en 2816, fuera del top 5. El ganador real (`paso_95M`) da 2866 con 30 episodios.

## H4 — 10 episodios no bastan para seleccionar por máximo de 5 (sección 2.3)

En la etapa 1 (10 episodios), `paso_45M` lideraba el ranking por E[máx de 5] con
3128. En la etapa 2 (30 episodios) cayó a 2834 y su media resultó 2370 con
desviación 774: el 3515 de la etapa 1 había sido un episodio afortunado.

`paso_95M` ganó **ambos** rankings en la etapa 2 (media 2866, E[máx5] 2974), lo
que elimina la necesidad de apostar a cuál métrica usa el enunciado.

## H5 — Un bug invalidó IT-04, y la revisión de código lo encontró (sección 2.3)

**Primera versión (inválida).** `ppo_02` retomó desde los 100M de `ppo_01` con un
config que declaraba `learning_rate: 1.0e-4` constante. Tras 60M pasos adicionales
terminó en 2,829, y se concluyó que "`ppo_01` ya había convergido".

**Esa conclusión era falsa.** Una revisión de código posterior detectó que
`Algo.load()` de Stable-Baselines3 restaura los *schedules serializados* del modelo
guardado, y que los hiperparámetros del config sólo se aplicaban en la rama del
modelo nuevo. El config se ignoraba en silencio. Los valores reales del log:

| Hiperparámetro | Configurado | Real |
|---|---|---|
| `learning_rate` | 1.0e-4 constante | 9.37e-05 → **2.8e-09** |
| `clip_range` | (heredado) | 0.0375 → **1.12e-06** |

Con `clip_range` en 1e-06 la política prácticamente no podía actualizarse. El
experimento **no midió lo que decía medir**: no hubo mejora porque el agente no
entrenó, no porque ya hubiera convergido.

**Corrección.** Se añadió `custom_objects` en la carga y una función que aplica
explícitamente los hiperparámetros del config al modelo cargado, informando cuáles
no son aplicables tras la construcción (por ejemplo `n_steps`, que dimensiona el
buffer de rollout). IT-04 se re-ejecutó como `ppo_03` con `learning_rate=1e-4` y
`clip_range=0.1` verificados en el log.

**Lección metodológica:** un experimento de RL puede fallar silenciosamente y
producir un número plausible. El resultado "sin mejora" era creíble y encajaba con
la hipótesis de convergencia, que es exactamente lo que lo hacía peligroso.

## H6 — La señal de recompensa es dispersa (sección 2.1)

Medido sobre 2,928 pasos de 5 episodios aleatorios en el entorno de evaluación:

- **1.95%** de los pasos otorga recompensa (uno cada ~51 decisiones).
- Valores observados: 5, 10, 15, 20, 25, 30 (filas de invasores) y 200 (nave nodriza).
- Recompensa media por paso: 0.360. Máxima en un paso: 200.
- No existe recompensa negativa: perder una vida no penaliza, sólo acorta el episodio.

Un acierto a la nave nodriza vale lo mismo que 40 invasores de la primera fila. Esa
asimetría es la que justifica el **reward clipping a `sign(r)`** durante el
entrenamiento: sin recortar, el error temporal-diferencia de un UFO es 40 veces el de
un invasor común y desestabiliza el gradiente. El puntaje se reporta siempre crudo.
