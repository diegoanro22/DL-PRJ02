from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

FuncionAgente = Callable[[Any, gym.Env], Any]
SEMILLA_BASE = 2026


def crear_entorno(
    nombre_entorno: str,
    video_folder: str | Path | None = None,
    episode_trigger: Callable[[int], bool] | None = None,
    name_prefix: str = "rl-video",
    **kwargs: Any,
) -> gym.Env:
    if not nombre_entorno:
        raise ValueError("nombre_entorno no puede estar vacío")

    if video_folder is None:
        return gym.make(nombre_entorno, **kwargs)

    render_mode = kwargs.pop("render_mode", "rgb_array")
    if render_mode != "rgb_array":
        raise ValueError("La grabación requiere render_mode='rgb_array'")

    carpeta = Path(video_folder)
    carpeta.mkdir(parents=True, exist_ok=True)
    entorno = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)
    disparador = episode_trigger if episode_trigger is not None else lambda _: True
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Overwriting existing videos.*", category=UserWarning
        )
        return gym.wrappers.RecordVideo(
            entorno,
            video_folder=str(carpeta),
            episode_trigger=disparador,
            name_prefix=name_prefix,
            disable_logger=True,
        )


def agente_aleatorio(observation: Any, env: gym.Env) -> Any:
    del observation
    return env.action_space.sample()


def agente_regla_simple(observation: Any, env: gym.Env) -> int:
    del observation
    entorno_base = env.unwrapped
    if not hasattr(entorno_base, "get_action_meanings"):
        raise ValueError("agente_regla_simple requiere un entorno Atari/ALE")

    significados = list(entorno_base.get_action_meanings())
    requeridas = {"LEFTFIRE", "RIGHTFIRE"}
    if not requeridas.issubset(significados):
        raise ValueError(
            "El entorno debe ofrecer LEFTFIRE y RIGHTFIRE; "
            f"acciones disponibles: {significados}"
        )

    frame = int(entorno_base.ale.getEpisodeFrameNumber())
    accion = "RIGHTFIRE" if (frame // 120) % 2 == 0 else "LEFTFIRE"
    return significados.index(accion)


def ejecutar_episodio(
    env: gym.Env,
    funcion_agente: FuncionAgente,
    max_steps: int = 10_000,
    seed: int | None = None,
) -> dict[str, int | float | bool | None]:
    if max_steps <= 0:
        raise ValueError("max_steps debe ser mayor que cero")

    observation, _ = env.reset(seed=seed)
    if seed is not None:
        env.action_space.seed(seed)

    pasos = 0
    recompensa_total = 0.0
    terminated = False
    truncated = False

    while pasos < max_steps and not (terminated or truncated):
        action = funcion_agente(observation, env)
        observation, reward, terminated, truncated, _ = env.step(action)
        recompensa_total += float(reward)
        pasos += 1

    limite_alcanzado = pasos == max_steps and not (terminated or truncated)
    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "limite_pasos_alcanzado": limite_alcanzado,
        "seed": seed,
    }


def generar_video_agente(
    nombre_entorno: str,
    funcion_agente: FuncionAgente,
    video_folder: str | Path,
    name_prefix: str,
    n_episodios: int = 1,
    *,
    base_seed: int = SEMILLA_BASE,
    max_steps: int = 10_000,
    fabrica_entorno: Callable[..., gym.Env] | None = None,
    **make_kwargs: Any,
) -> dict[str, list[Any]]:
    """Graba `n_episodios` del agente y devuelve las rutas de video y las métricas.

    `fabrica_entorno` permite inyectar una construcción de entorno distinta a
    `crear_entorno` (por ejemplo, la del entorno de evaluación con el
    preprocesamiento Atari). Recibe los mismos argumentos que `crear_entorno` y
    debe devolver un entorno Gymnasium ya envuelto con `RecordVideo`.
    """
    if n_episodios <= 0:
        raise ValueError("n_episodios debe ser mayor que cero")

    carpeta = Path(video_folder)
    carpeta.mkdir(parents=True, exist_ok=True)
    patron = f"{name_prefix}*.mp4"
    estado_previo = {
        ruta.resolve(): (ruta.stat().st_mtime_ns, ruta.stat().st_size)
        for ruta in carpeta.glob(patron)
    }

    constructor = fabrica_entorno if fabrica_entorno is not None else crear_entorno
    env = constructor(
        nombre_entorno,
        video_folder=carpeta,
        episode_trigger=lambda episodio: episodio < n_episodios,
        name_prefix=name_prefix,
        **make_kwargs,
    )
    metricas: list[dict[str, int | float | bool | None]] = []
    try:
        for episodio in range(n_episodios):
            resultado = ejecutar_episodio(
                env,
                funcion_agente,
                max_steps=max_steps,
                seed=base_seed + episodio,
            )
            resultado["episodio"] = episodio + 1
            metricas.append(resultado)
    finally:
        env.close()

    videos: list[str] = []
    for ruta in sorted(carpeta.glob(patron)):
        ruta_resuelta = ruta.resolve()
        estado_actual = (ruta.stat().st_mtime_ns, ruta.stat().st_size)
        if estado_previo.get(ruta_resuelta) != estado_actual:
            videos.append(str(ruta_resuelta))

    if len(videos) < n_episodios:
        raise RuntimeError(
            f"Se esperaban {n_episodios} videos y se detectaron {len(videos)}"
        )

    return {"videos": videos, "metricas": metricas}


__all__ = [
    "agente_aleatorio",
    "agente_regla_simple",
    "crear_entorno",
    "ejecutar_episodio",
    "generar_video_agente",
]
