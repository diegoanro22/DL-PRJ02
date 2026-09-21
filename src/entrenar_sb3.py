"""Entrenamiento de agentes SB3 sobre ALE/SpaceInvaders-v5.

Uso:
    python src/entrenar_sb3.py --config configs/ppo.yaml --run-id ppo_01
    python src/entrenar_sb3.py --config configs/ppo.yaml --run-id ppo_01 --resume
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import yaml

gym.register_envs(ale_py)  # SB3 llama a gym.make por dentro

sys.path.insert(0, str(Path(__file__).parent))
from entorno import crear_entorno_entrenamiento, crear_entorno_evaluacion  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def _parsear_schedule(valor):
    """'lin_2.5e-4' -> schedule lineal, como en RL-Zoo."""
    if isinstance(valor, str) and valor.startswith("lin_"):
        inicial = float(valor[4:])

        def schedule(progreso_restante: float) -> float:
            return progreso_restante * inicial

        return schedule
    return valor


_SCHEDULES = ("learning_rate", "clip_range", "clip_range_vf")
# Solo se pueden fijar al construir el modelo.
_SOLO_CONSTRUCCION = ("policy_kwargs", "buffer_size", "n_steps", "n_envs",
                      "optimize_memory_usage", "replay_buffer_kwargs")


def _objetos_a_reemplazar(hp: dict) -> dict:
    """Lo que `Algo.load` debe tomar del config y no del .zip."""
    return {k: v for k, v in hp.items() if k in _SCHEDULES}


def _aplicar_hiperparametros(model, hp: dict) -> None:
    """Pisa en el modelo cargado lo que declara el config.

    Sin esto un `--resume` con config nuevo seguiría usando los del run viejo.
    """
    from stable_baselines3.common.utils import get_schedule_fn

    aplicados, ignorados = [], []
    for k, v in hp.items():
        if k in _SOLO_CONSTRUCCION:
            ignorados.append(k)
            continue
        if k in _SCHEDULES:
            setattr(model, f"{k}_schedule" if k == "learning_rate" else k, get_schedule_fn(v))
            if k == "learning_rate":
                model.learning_rate = v
            aplicados.append(k)
        elif hasattr(model, k):
            setattr(model, k, v)
            aplicados.append(k)
        else:
            ignorados.append(k)
    print(f"[resume] hiperparametros aplicados: {', '.join(sorted(aplicados))}")
    if ignorados:
        print(f"[resume] NO aplicables tras la carga (requieren modelo nuevo): "
              f"{', '.join(sorted(ignorados))}")


def _construir_algo(nombre: str):
    if nombre == "ppo":
        from stable_baselines3 import PPO

        return PPO
    if nombre == "dqn":
        from stable_baselines3 import DQN

        return DQN
    if nombre == "a2c":
        from stable_baselines3 import A2C

        return A2C
    if nombre == "qrdqn":
        from sb3_contrib import QRDQN

        return QRDQN
    raise ValueError(f"Algoritmo desconocido: {nombre}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="Sobrescribe el del config (para pruebas cortas)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    n_envs = int(cfg["n_envs"])
    total = int(args.total_timesteps or cfg["total_timesteps"])

    dir_run = RAIZ / "artifacts" / "checkpoints" / args.run_id
    dir_log = RAIZ / "artifacts" / "logs" / args.run_id
    dir_run.mkdir(parents=True, exist_ok=True)
    dir_log.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, dir_run / "config.yaml")

    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor

    env = crear_entorno_entrenamiento(
        n_envs=n_envs, seed=2026, monitor_dir=str(dir_log / "monitor")
    )
    # Gymnasium suelto: 3 vidas y recompensa cruda.
    env_eval = Monitor(crear_entorno_evaluacion(), filename=str(dir_log / "eval_monitor"))

    # Los callbacks cuentan llamadas a _on_step(), una por paso del VecEnv, o sea
    # cada n_envs transiciones. En el config van en transiciones.
    eval_cada = int(cfg.get("eval_cada", 1_000_000))
    save_cada = int(cfg.get("save_cada", 1_000_000))
    eval_freq = max(eval_cada // n_envs, 1)
    save_freq = max(save_cada // n_envs, 1)
    print(f"[freqs] n_envs={n_envs} | eval cada {eval_freq*n_envs:,} transiciones "
          f"(_on_step={eval_freq:,}) | ckpt cada {save_freq*n_envs:,} (_on_step={save_freq:,})")

    callbacks = [
        # Sin save_replay_buffer, un resume off-policy arranca sin memoria.
        CheckpointCallback(save_freq=save_freq, save_path=str(dir_run),
                           name_prefix="paso", verbose=1,
                           save_replay_buffer=cfg["algo"] in ("dqn", "qrdqn")),
        # best_model se guarda por recompensa media, no por el máximo.
        EvalCallback(env_eval, best_model_save_path=str(dir_run),
                     log_path=str(dir_log), eval_freq=eval_freq,
                     n_eval_episodes=5, deterministic=True, verbose=1),
    ]

    Algo = _construir_algo(cfg["algo"])
    hp = {k: _parsear_schedule(v) for k, v in cfg["hiperparametros"].items()}

    ruta_ultimo = dir_run / "ultimo.zip"
    if args.resume and ruta_ultimo.exists():
        print(f"[resume] cargando {ruta_ultimo}")
        # custom_objects para que los schedules salgan del config y no del .zip.
        model = Algo.load(ruta_ultimo, env=env, tensorboard_log=str(dir_log),
                          custom_objects=_objetos_a_reemplazar(hp))
        _aplicar_hiperparametros(model, hp)
        reset_num_timesteps = False
        ruta_buffer = dir_run / "ultimo_replay_buffer.pkl"
        if hasattr(model, "replay_buffer"):
            if ruta_buffer.exists():
                model.load_replay_buffer(ruta_buffer)
                print(f"[resume] replay buffer restaurado: {model.replay_buffer.size():,} transiciones")
            else:
                print("[resume] AVISO: no hay replay buffer guardado; se reanuda con memoria vacia")
    else:
        model = Algo(cfg["politica"], env, verbose=1,
                     tensorboard_log=str(dir_log), seed=2026, **hp)
        reset_num_timesteps = True

    print(f"[device] {model.device}")
    try:
        model.learn(total_timesteps=total, callback=callbacks,
                    reset_num_timesteps=reset_num_timesteps,
                    tb_log_name=args.run_id, progress_bar=False)
    finally:
        model.save(ruta_ultimo)
        print(f"[guardado] {ruta_ultimo}")
        if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
            model.save_replay_buffer(dir_run / "ultimo_replay_buffer.pkl")
            print(f"[guardado] replay buffer ({model.replay_buffer.size():,} transiciones)")
        env.close()
        env_eval.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
