"""Selección de checkpoint en dos etapas.

Primero un filtro con pocos episodios sobre todos los checkpoints, después
muchos episodios sobre los finalistas. Reporta media y E[máx de 5] porque el
enunciado no deja claro cuál de las dos define el ranking.

Las semillas son distintas a las del entrenamiento y las evaluaciones periódicas.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)
sys.path.insert(0, str(Path(__file__).parent))
from ale_interaccion import ejecutar_episodio  # noqa: E402
from entorno import crear_entorno_evaluacion, hacer_agente_sb3  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def e_max_de_k(puntajes: np.ndarray, k: int = 5, n: int = 20000, semilla: int = 0) -> float:
    """E[máx de k episodios], por bootstrap sobre los puntajes observados."""
    rng = np.random.default_rng(semilla)
    return float(rng.choice(puntajes, size=(n, k), replace=True).max(axis=1).mean())


def evaluar(ruta_modelo: Path, algo: str, n_ep: int, semilla_base: int,
            deterministic: bool = True) -> np.ndarray:
    if algo == "qrdqn":
        from sb3_contrib import QRDQN as Algo
    else:
        from stable_baselines3 import PPO as Algo
    model = Algo.load(ruta_modelo, device="cuda")
    env = crear_entorno_evaluacion()
    agente = hacer_agente_sb3(model, deterministic=deterministic)
    try:
        return np.array([
            ejecutar_episodio(env, agente, max_steps=27_000, seed=semilla_base + i)["recompensa_total"]
            for i in range(n_ep)
        ])
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--episodios", type=int, default=10)
    ap.add_argument("--semilla-base", type=int, default=9000)
    ap.add_argument("--solo", nargs="*", default=None, help="rutas concretas en vez de todos")
    ap.add_argument("--estocastico", action="store_true")
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    dir_ck = RAIZ / "artifacts" / "checkpoints" / args.run_id
    if args.solo:
        rutas = [Path(p) for p in args.solo]
    else:
        rutas = sorted(dir_ck.glob("paso_*.zip"),
                       key=lambda p: int(p.stem.split("_")[1]))
        for extra in ("best_model.zip", "ultimo.zip"):
            if (dir_ck / extra).exists():
                rutas.append(dir_ck / extra)

    print(f"Evaluando {len(rutas)} checkpoints x {args.episodios} episodios "
          f"(determinista={not args.estocastico})\n")
    print(f"{'checkpoint':<28} {'media':>7} {'std':>6} {'medi':>6} {'max':>6} "
          f"{'min':>6} {'E[max5]':>8} {'seg':>5}")
    filas = []
    for ruta in rutas:
        t0 = time.time()
        p = evaluar(ruta, args.algo, args.episodios, args.semilla_base,
                    deterministic=not args.estocastico)
        fila = {
            "checkpoint": ruta.name, "ruta": str(ruta),
            "puntajes": [float(x) for x in p],
            "media": float(p.mean()), "std": float(p.std()),
            "mediana": float(np.median(p)), "max": float(p.max()),
            "min": float(p.min()), "e_max5": e_max_de_k(p),
        }
        filas.append(fila)
        print(f"{ruta.name:<28} {fila['media']:>7.0f} {fila['std']:>6.0f} "
              f"{fila['mediana']:>6.0f} {fila['max']:>6.0f} {fila['min']:>6.0f} "
              f"{fila['e_max5']:>8.0f} {time.time()-t0:>5.0f}")

    print("\n=== RANKING por MEDIA ===")
    for i, f in enumerate(sorted(filas, key=lambda x: -x["media"])[:5], 1):
        print(f"  {i}. {f['checkpoint']:<28} media {f['media']:>6.0f}  E[max5] {f['e_max5']:>6.0f}")
    print("=== RANKING por E[max de 5] ===")
    for i, f in enumerate(sorted(filas, key=lambda x: -x["e_max5"])[:5], 1):
        print(f"  {i}. {f['checkpoint']:<28} E[max5] {f['e_max5']:>6.0f}  media {f['media']:>6.0f}")

    salida = Path(args.salida or RAIZ / "artifacts" / "evaluaciones" / f"embudo_{args.run_id}.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(filas, indent=2))
    print(f"\nguardado en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
