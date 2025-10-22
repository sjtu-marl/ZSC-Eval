from pathlib import Path
from typing import List

ALG_SPECS = {
    "FCP":    (3, "rnn",  "fcp/s2",      "fcp-S2-s24"),
    "MEP":    (3, "rnn",  "mep/s2",      "mep-S2-s24"),
    "TrajeDi":(3, "rnn",  "traj/s2",     "traj-S2-s24"),
    "HSP":    (3, "rnn",  "hsp/s2",      "hsp-S2-s24"),
    "SP":     (3, "mlp",  "fcp/s1",      "sp-S1-s15"),
    "E3T":    (3, "mlp",  "e3t/s1",      "e3t"),
    "COLE":   (3, "rnn",  "cole/s2",     "cole-S2-s50"),
}


def policy_config_path(kind: str, prefix: str) -> str:
    return f"{prefix}/policy_config/{'rnn' if kind=='rnn' else 'mlp'}_policy_config.pkl"


def actor_path(algo: str, idx: int, prefix: str, stub: str, tag: str) -> str:
    if algo == "SP":
        return f"{prefix}/{stub}/{tag}/sp{idx}_final_actor.pt"
    if algo == "E3T":
        return f"{prefix}/{stub}/{tag}/{idx}.pt"
    return f"{prefix}/{stub}/{tag}/{idx}.pt"


def block_name(algo: str, idx: int) -> str:
    head = {"FCP":"fcp","MEP":"mep","TrajeDi":"traj","HSP":"hsp","SP":"sp","E3T":"e3t","COLE":"cole"}[algo]
    return f"{head}{idx}"


def make_yaml_for_prefix(prefix: str) -> str:
    lines = []
    for algo, (cnt, kind, stub, tag) in ALG_SPECS.items():
        for i in range(1, cnt+1):
            name = block_name(algo, i)
            cfg  = policy_config_path(kind, prefix)
            actor = actor_path(algo, i, prefix, stub, tag)
            lines += [
                f"{name}:",
                f"    policy_config_path: {cfg}",
                f"    featurize_type: ppo",
                f"    algo: {algo}",
                f"    model_path:",
                f"        actor: {actor}",
            ]
    return "\n".join(lines) + "\n"


def save_benchmarks(
    maps: List[str],
    output_dir: str = "config",
    make_m_variant: bool = True,
):
    cfg_dir = Path(output_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    for m in maps:
        txt = make_yaml_for_prefix(m)
        (cfg_dir / f"{m}_benchmark.yaml").write_text(txt, encoding="utf-8")

        if make_m_variant:
            txt_m = make_yaml_for_prefix(f"{m}_m")
            (cfg_dir / f"{m}_m_benchmark.yaml").write_text(txt_m, encoding="utf-8")


if __name__ == "__main__":
    maps = ["random0_medium", "random1", "random3", "small_corridor", "unident_s", "random1_m", "random3_m"]
    save_benchmarks(maps, output_dir="config", make_m_variant=True)
    print("Saved to ./config/<map>_benchmark.yaml and <map>_m_benchmark.yaml")