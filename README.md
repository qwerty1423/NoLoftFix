# NOLoftFix

This project is an unofficial community modification and is not affiliated with, sponsored by, or endorsed by Shockfront Studios Pty Ltd. Original Nuclear Option assets, vehicle designs, audio, and code are Copyright (c) 2026 Shockfront Studios Pty Ltd. All rights reserved. Nuclear Option and Shockfront Studios are trademarks or registered trademarks of Shockfront Studios Pty Ltd. Original mod content and all other trademarks belong to their respective owners.

---

warning - LLMs were used

some files were omitted from the repo

code quality may be low

to use... compile it yourself I guess.

You can use sim/grid_dymos.py to run the optimiser, it will make loft_tables.json for you. The process is something like this, I think? replace --jobs 15 with around twice the number of your cpu cores.

you will also need to run parse_unity.py to get a copy of coeffs_aam2.json.

```bash
uv venv
source .venv/bin/activate
uv pip install -r sim/requirements.txt
cd sim

python3 parse_unity.py --prefab ../txt/AAM2.txt --assets ../txt/GameAssets.txt --out coeffs_aam2.json

OMP_NUM_THREADS=1 python3 grid_dymos.py --grid grids/scythe.json --jobs 15 --max-tasks 300 --timeout 100000

python3 grid_dymos.py --grid grids/scythe.json --merge
```

and then you can copy the table to mod/Tables

I have provided a loft table I have generated, but currently only at 250 max iterations so it isn't great.

might break the scimitar, it is only good for the scythe currently? probably just the scimitar being bad because of nerfs though.

based on <https://doi.org/10.82124/CEAS-GNC-2026-016> but vibecoded so... again... the implementation is probably not great
