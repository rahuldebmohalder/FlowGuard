import multiprocessing as mp
import time, json, sys
from pathlib import Path

def worker(source, q):
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.parsers.solidity_parser import RegexParser
        RegexParser().parse(source)
        q.put("ok")
    except Exception as e:
        q.put(f"err:{e}")

def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from src.parsers.dataset_loader import SCVulnLoader, SmartBugsLoader

    sb = SmartBugsLoader().load()
    sc_vuln = SCVulnLoader().load(only_enum=True, max_source_chars=20000)
    all_contracts = sb + sc_vuln
    print(f"Testing {len(all_contracts)} contracts with 4s hard timeout...")

    blacklist = []
    for i, rec in enumerate(all_contracts):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=worker, args=(rec.source_code, q))
        t0 = time.time()
        p.start()
        p.join(4)
        if p.is_alive():
            p.terminate()
            p.join(1)
            if p.is_alive():
                p.kill()
            print(f"  [{i+1:3d}/{len(all_contracts)}] {rec.contract_id}: HANG — BLACKLISTED")
            blacklist.append(rec.contract_id)
        else:
            elapsed = time.time() - t0
            if elapsed > 1.0:
                print(f"  [{i+1:3d}/{len(all_contracts)}] {rec.contract_id}: slow ({elapsed:.1f}s)")

    Path("src/parsers/blacklist.json").write_text(json.dumps(blacklist, indent=2))
    print(f"\nDone. Blacklisted {len(blacklist)}/{len(all_contracts)} contracts")
    print(f"Saved to blacklist.json")

if __name__ == "__main__":
    mp.freeze_support()
    main()