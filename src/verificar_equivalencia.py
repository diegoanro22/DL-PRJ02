"""Compara las observaciones de las dos rutas de entorno.

Si no coinciden, el agente juega con una entrada distinta a la que vio
entrenando y el puntaje cae sin que nada falle.

Para que la comparación sea determinista se apagan las dos fuentes de azar:
`noop_max=0` y `repeat_action_probability=0.0`. En uso normal van activadas.
"""

from __future__ import annotations

import sys

import ale_py
import gymnasium as gym
import numpy as np

sys.path.insert(0, "src")

gym.register_envs(ale_py)  # antes de make_atari_env


def _ruta_sb3(acciones: list[int]) -> list[np.ndarray]:
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        VecFrameStack,
        VecTransposeImage,
    )

    env = make_atari_env(
        "ALE/SpaceInvaders-v5",
        n_envs=1,
        seed=2026,
        env_kwargs={"frameskip": 1, "repeat_action_probability": 0.0},
        wrapper_kwargs={
            "noop_max": 0,
            "frame_skip": 4,
            "screen_size": 84,
            "terminal_on_life_loss": True,
            "clip_reward": True,
        },
        vec_env_cls=DummyVecEnv,
    )
    env = VecFrameStack(env, n_stack=4)
    # SB3 mete esto solo al construir el modelo: (84,84,4) -> (4,84,84).
    env = VecTransposeImage(env)
    obs = env.reset()
    salida = [np.array(obs[0])]
    for a in acciones:
        obs, _, _, _ = env.step(np.array([a]))
        salida.append(np.array(obs[0]))
    env.close()
    return salida


def _ruta_gym(acciones: list[int]) -> list[np.ndarray]:
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    from entorno import _ResetConFire

    env = gym.make("ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.0)
    env = AtariPreprocessing(
        env, noop_max=0, frame_skip=4, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
    )
    env = _ResetConFire(env)
    env = FrameStackObservation(env, 4, padding_type="zero")
    obs, _ = env.reset(seed=2026)
    salida = [np.array(obs)]
    for a in acciones:
        obs, _, _, _, _ = env.step(a)
        salida.append(np.array(obs))
    env.close()
    return salida


def main() -> int:
    acciones = [0, 1, 2, 3, 4, 5, 2, 4, 1, 3]
    a = _ruta_sb3(acciones)
    b = _ruta_gym(acciones)

    print(f"{'paso':>5} {'shape sb3':>14} {'shape gym':>14} {'iguales':>8} {'dif_max':>8} {'dif_media':>10}")
    todas_iguales = True
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape:
            print(f"{i:>5} {str(x.shape):>14} {str(y.shape):>14} {'SHAPE!':>8}")
            todas_iguales = False
            continue
        igual = np.array_equal(x, y)
        d = np.abs(x.astype(int) - y.astype(int))
        print(f"{i:>5} {str(x.shape):>14} {str(y.shape):>14} {str(igual):>8} "
              f"{d.max():>8} {d.mean():>10.3f}")
        todas_iguales &= igual

    print()
    if todas_iguales:
        print("RESULTADO: las dos rutas son IDÉNTICAS.")
        return 0
    print("RESULTADO: LAS RUTAS DIVERGEN. No entrenar hasta corregirlo.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
