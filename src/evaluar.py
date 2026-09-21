"""Evaluación del agente entrenado.

    python src/evaluar.py --agente artifacts/agente_final.zip --episodios 5 --video

Carga los pesos y el yaml de wrappers de al lado, corre N episodios greedy,
imprime la tabla de puntajes y, si se pide, graba el video. Usa
`ejecutar_episodio` y `generar_video_agente` del laboratorio 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import yaml

gym.register_envs(ale_py)
sys.path.insert(0, str(Path(__file__).parent))
from ale_interaccion import ejecutar_episodio, generar_video_agente  # noqa: E402
from entorno import crear_entorno_evaluacion, hacer_agente_sb3  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def cargar(ruta_agente: Path):
    """Devuelve el modelo y el yaml de configuración que lo acompaña."""
    ruta_cfg = ruta_agente.with_suffix("").with_suffix(".config.yaml")
    if not ruta_cfg.exists():
        ruta_cfg = ruta_agente.parent / f"{ruta_agente.stem}.config.yaml"
    cfg = yaml.safe_load(ruta_cfg.read_text()) if ruta_cfg.exists() else {}
    algo = cfg.get("agente", {}).get("algoritmo", "ppo")
    if algo == "qrdqn":
        from sb3_contrib import QRDQN as Algo
    else:
        from stable_baselines3 import PPO as Algo
    return Algo.load(ruta_agente), cfg, ruta_cfg


def kwargs_desde_config(cfg: dict) -> dict:
    """Traduce el yaml de configuración a argumentos de crear_entorno_evaluacion."""
    e = cfg.get("entorno", {})
    pre = e.get("preprocesamiento", {})
    fs = e.get("frame_stack", {})
    kw = {"nombre_entorno": e.get("id", "ALE/SpaceInvaders-v5")}
    if "kwargs_ale" in e:
        kw["kwargs_ale"] = e["kwargs_ale"]
    for origen, destino in (("noop_max", "noop_max"), ("frame_skip", "frame_skip"),
                            ("screen_size", "screen_size")):
        if origen in pre:
            kw[destino] = pre[origen]
    if "n" in fs:
        kw["n_stack"] = fs["n"]
    if "padding_type" in fs:
        kw["padding_type"] = fs["padding_type"]
    if "fire_reset" in e:
        kw["fire_reset"] = e["fire_reset"]
    # En evaluación no van, así que un yaml que los pida está mal.
    if pre.get("terminal_on_life_loss", False):
        raise ValueError("El sidecar pide terminal_on_life_loss=True en evaluacion: "
                         "el episodio debe ser las 3 vidas completas")
    if e.get("clip_reward", False):
        raise ValueError("El sidecar pide clip_reward=True en evaluacion: "
                         "el puntaje se reporta crudo")
    return kw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agente", default=str(RAIZ / "artifacts" / "agente_final.zip"))
    ap.add_argument("--episodios", type=int, default=5)
    ap.add_argument("--semilla-base", type=int, default=2026)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--estocastico", action="store_true")
    args = ap.parse_args()

    ruta = Path(args.agente)
    model, cfg, ruta_cfg = cargar(ruta)
    determinista = not args.estocastico

    print(f"Agente      : {ruta}")
    print(f"Config      : {ruta_cfg if ruta_cfg.exists() else '(no encontrada, usando defaults)'}")
    kw_env = kwargs_desde_config(cfg)
    print(f"Entorno     : {kw_env['nombre_entorno']}")
    print(f"Wrappers    : frame_skip={kw_env.get('frame_skip', 4)} "
          f"screen={kw_env.get('screen_size', 84)} stack={kw_env.get('n_stack', 4)} "
          f"fire_reset={kw_env.get('fire_reset', True)} "
          f"padding={kw_env.get('padding_type', 'zero')}")
    print(f"Politica    : {'determinista' if determinista else 'estocastica'}")
    print(f"Episodios   : {args.episodios}  (semillas {args.semilla_base}..{args.semilla_base+args.episodios-1})\n")

    agente = hacer_agente_sb3(model, deterministic=determinista)

    if args.video:
        carpeta = RAIZ / "videos"
        fabrica = lambda nombre, **kw: crear_entorno_evaluacion(
            **{**kw_env, **kw, "nombre_entorno": kw_env["nombre_entorno"]})
        res = generar_video_agente(
            kw_env["nombre_entorno"],
            agente, video_folder=carpeta, name_prefix="agente_final",
            n_episodios=args.episodios, base_seed=args.semilla_base,
            max_steps=27_000, fabrica_entorno=fabrica,
        )
        puntajes = np.array([m["recompensa_total"] for m in res["metricas"]])
        pasos = [m["pasos"] for m in res["metricas"]]
        videos = res["videos"]
    else:
        env = crear_entorno_evaluacion(**kw_env)
        met = [ejecutar_episodio(env, agente, max_steps=27_000, seed=args.semilla_base + i)
               for i in range(args.episodios)]
        env.close()
        puntajes = np.array([m["recompensa_total"] for m in met])
        pasos = [m["pasos"] for m in met]
        videos = []

    print(f"{'episodio':>9} {'puntaje':>9} {'pasos':>8}")
    for i, (p, s) in enumerate(zip(puntajes, pasos), 1):
        print(f"{i:>9} {p:>9.0f} {s:>8}")
    print("-" * 28)
    print(f"{'MEDIA':>9} {puntajes.mean():>9.1f}")
    print(f"{'MAXIMO':>9} {puntajes.max():>9.0f}")
    print(f"{'minimo':>9} {puntajes.min():>9.0f}")
    print(f"{'mediana':>9} {np.median(puntajes):>9.1f}")
    print(f"{'std':>9} {puntajes.std():>9.1f}")

    for v in videos:
        print(f"\nvideo: {v}")

    salida = RAIZ / "artifacts" / "evaluaciones" / "evaluacion_final.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({
        "agente": str(ruta), "episodios": args.episodios,
        "determinista": determinista, "semilla_base": args.semilla_base,
        "puntajes": [float(x) for x in puntajes],
        "media": float(puntajes.mean()), "max": float(puntajes.max()),
        "min": float(puntajes.min()), "std": float(puntajes.std()),
        "mediana": float(np.median(puntajes)), "videos": videos,
    }, indent=2))
    print(f"\nresultados: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
