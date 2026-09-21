"""Construcción del entorno de Space Invaders.

Entrenamiento usa el VecEnv de SB3 (varios entornos en paralelo, API de 4
elementos). Evaluación y video usan un entorno Gymnasium suelto, que es lo que
esperan las funciones del laboratorio 5.

Ambos caminos tienen que dar la misma observación (4,84,84); verificar_equivalencia.py
lo comprueba.
"""

from __future__ import annotations

from typing import Any

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

ENTORNO = "ALE/SpaceInvaders-v5"

# v5 ya trae frameskip=4 y los wrappers de Atari aplican otro 4; en 1 para no
# acumular un salto de 16.
KWARGS_ALE: dict[str, Any] = {
    "frameskip": 1,
    "repeat_action_probability": 0.25,  # sticky actions, default de v5
}

NOOP_MAX = 30
FRAME_SKIP = 4
TAMANO_PANTALLA = 84
N_STACK = 4


def crear_entorno_entrenamiento(
    n_envs: int = 8,
    seed: int = 0,
    monitor_dir: str | None = None,
):
    """VecEnv con vidas episódicas y recompensa recortada."""
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecFrameStack,
    )

    env = make_atari_env(
        ENTORNO,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        env_kwargs=dict(KWARGS_ALE),
        wrapper_kwargs={
            "noop_max": NOOP_MAX,
            "frame_skip": FRAME_SKIP,
            "screen_size": TAMANO_PANTALLA,
            "terminal_on_life_loss": True,
            "clip_reward": True,
        },
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
    )
    return VecFrameStack(env, n_stack=N_STACK)


def crear_entorno_evaluacion(
    nombre_entorno: str = ENTORNO,
    video_folder: str | None = None,
    episode_trigger: Any = None,
    name_prefix: str = "agente",
    *,
    fire_reset: bool = True,
    kwargs_ale: dict[str, Any] | None = None,
    noop_max: int = NOOP_MAX,
    frame_skip: int = FRAME_SKIP,
    screen_size: int = TAMANO_PANTALLA,
    n_stack: int = N_STACK,
    padding_type: str = "zero",
    **_ignorados: Any,
) -> gym.Env:
    """Entorno de un solo proceso para evaluar y grabar.

    A diferencia del de entrenamiento, el episodio son las 3 vidas y la
    recompensa va sin recortar. La firma imita a `crear_entorno` del laboratorio 5
    para poder usarse como `fabrica_entorno` en `generar_video_agente`.
    """
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    render_mode = "rgb_array" if video_folder is not None else None
    env = gym.make(nombre_entorno, render_mode=render_mode,
                   **(kwargs_ale if kwargs_ale is not None else KWARGS_ALE))

    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )

    # AtariPreprocessing no trae el FireReset que SB3 sí aplica.
    if fire_reset:
        env = _ResetConFire(env)

    # SB3 rellena el stack con ceros; el default de Gymnasium repite el primer frame.
    env = FrameStackObservation(env, n_stack, padding_type=padding_type)

    if video_folder is not None:
        from pathlib import Path

        Path(video_folder).mkdir(parents=True, exist_ok=True)
        disparador = episode_trigger if episode_trigger is not None else (lambda _: True)
        # Afuera de todo: render() devuelve el RGB original, no el 84x84 del agente.
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=disparador,
            name_prefix=name_prefix,
            disable_logger=True,
        )

    return env


class _ResetConFire(gym.Wrapper):
    """Equivalente al FireResetEnv de SB3."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        significados = env.unwrapped.get_action_meanings()
        if significados[1] != "FIRE" or len(significados) < 3:
            raise ValueError(f"FireResetEnv requiere FIRE en el índice 1: {significados}")

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        # Son dos pasos, no uno.
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        return obs, {}


def hacer_agente_sb3(model, deterministic: bool = True):
    """Envuelve un modelo de SB3 en la firma que espera `ejecutar_episodio`."""

    def agente(observation: Any, env: gym.Env) -> int:
        del env
        accion, _ = model.predict(observation, deterministic=deterministic)
        return int(accion)

    return agente


__all__ = [
    "ENTORNO",
    "KWARGS_ALE",
    "crear_entorno_entrenamiento",
    "crear_entorno_evaluacion",
    "hacer_agente_sb3",
]
