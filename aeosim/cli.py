"""Command line: python -m aeosim <command> ...

  run       configs/pilot.yaml [--limit N]       run an experiment (add --allow-dirty to run uncommitted code)
  analyze   runs/pilot/<commit> [--no-firm-fe]   funnel + conditional logit report
  seller    configs/seller.yaml [--limit N]      paired before/after seller experiment
  preview   configs/pilot.yaml --scenario 3      print one scenario's brief, market and a page
  serve     configs/pilot.yaml --scenario 3      browse one scenario's mock web at localhost:8000
  briefs    --category freight_trucking --n 60   export briefs for real-AI-search validation
  audit     urls.txt --out audit.csv             detect AEO features on real websites
"""

from __future__ import annotations

import argparse
from pathlib import Path

import os


def load_env(path: str = ".env") -> None:
    """Read KEY=value lines from .env into the environment (API keys)."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(prog="aeosim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--limit", type=int)
    r.add_argument("--allow-dirty", action="store_true")
    a = sub.add_parser("analyze"); a.add_argument("run_dir"); a.add_argument("--no-firm-fe", action="store_true")
    a.add_argument("--file", default="episodes.jsonl")
    s = sub.add_parser("seller"); s.add_argument("config"); s.add_argument("--limit", type=int)
    s.add_argument("--allow-dirty", action="store_true")
    rp = sub.add_parser("replay"); rp.add_argument("run_dir"); rp.add_argument("--scenario", type=int)
    rp.add_argument("--model"); rp.add_argument("--profile"); rp.add_argument("--file", default="episodes.jsonl")
    sm = sub.add_parser("summary"); sm.add_argument("run_dir"); sm.add_argument("--file", default="episodes.jsonl")
    pv = sub.add_parser("preview"); pv.add_argument("config"); pv.add_argument("--scenario", type=int, default=0)
    sv = sub.add_parser("serve"); sv.add_argument("config"); sv.add_argument("--scenario", type=int, default=0)
    sv.add_argument("--port", type=int, default=8000)
    b = sub.add_parser("briefs"); b.add_argument("--category", default="freight_trucking")
    b.add_argument("--n", type=int, default=60); b.add_argument("--seed", type=int, default=1)
    b.add_argument("--out", default="validation")
    au = sub.add_parser("audit"); au.add_argument("urls_file"); au.add_argument("--out", default="audit.csv")
    args = p.parse_args(argv)

    from .config import load_config
    from .provenance import DirtyRepoError
    try:
        _dispatch(args, load_config)
    except DirtyRepoError as e:
        raise SystemExit(f"Run stopped. {e}")


def _dispatch(args, load_config):
    def cfg_from(path):
        cfg = load_config(path)
        cfg["_allow_dirty"] = getattr(args, "allow_dirty", False)
        return cfg

    if args.cmd == "run":
        from .runner import run_experiment
        out = run_experiment(cfg_from(args.config), limit=args.limit)
        print(f"Done. Analyze with: python -m aeosim analyze {out.parent}")
    elif args.cmd == "analyze":
        from .analysis import analyze
        print(Path(analyze(args.run_dir, args.file, firm_fe=not args.no_firm_fe)).read_text())
    elif args.cmd in ("replay", "summary"):
        from .analysis import load_episodes
        from .present import pick_episode, replay, resolve_run, summary
        run = resolve_run(args.run_dir, args.file)
        if args.cmd == "summary":
            print(summary(run, args.file))
        else:
            print(replay(pick_episode(load_episodes(run / args.file), args.scenario, args.model, args.profile)))
    elif args.cmd == "seller":
        from .seller import run_seller
        print(Path(run_seller(cfg_from(args.config), limit=args.limit)).read_text())
    elif args.cmd in ("preview", "serve"):
        from .market import generate_pool
        from .scenario import build_scenario
        cfg = load_config(args.config)
        exp = cfg["experiment"]
        pool = generate_pool(exp["category"], exp["pool_size"], exp["corpus_seed"])
        sc = build_scenario(exp, cfg["features"], pool, args.scenario)
        web = sc.build_web(cfg["web"])
        if args.cmd == "preview":
            print("BRIEF:", sc.brief.text, "\n")
            for m in sc.market_record():
                on = [k for k, v in m["features"].items() if v]
                print(f"{m['firm_id']} {m['name']:<28} fit={m['fit']['full_fit']} features={on}")
            first = sc.firms[0]
            print(f"\nWhat the agent reads at https://{first.domain}/ :\n")
            print(web.fetch(f"https://{first.domain}/")["text"])
        else:
            from .serve import serve
            serve(web, sc, args.port)
    elif args.cmd == "briefs":
        from .validation import export_briefs
        print(export_briefs(args.category, args.n, args.seed, args.out))
    elif args.cmd == "audit":
        from .validation import audit_sites
        urls = [u.strip() for u in Path(args.urls_file).read_text().splitlines() if u.strip()]
        print(audit_sites(urls, args.out))


if __name__ == "__main__":
    main()
