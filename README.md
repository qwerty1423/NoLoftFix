# NOLoftFix

This project is an unofficial community modification and is not affiliated with, sponsored by, or endorsed by Shockfront Studios Pty Ltd. Original Nuclear Option assets, vehicle designs, audio, and code are Copyright (c) 2026 Shockfront Studios Pty Ltd. All rights reserved. Nuclear Option and Shockfront Studios are trademarks or registered trademarks of Shockfront Studios Pty Ltd. Original mod content and all other trademarks belong to their respective owners.

---

warning - LLMs were used

some files were omitted from the repo

code quality may be low

to use... compile it yourself I guess.

The mod only patches code that runs if you are host, so you can join vanilla games with it if you want, it won't do anything. If you are the host, then the mod will change the lofting behaviour.

May improve scythe performance. The scythe will now hit targets up to around 250 km as long as they are easy targets such as AI helicopters, by lofting properly. AI fighters can be hit around half the time at around 90 km. Effectiveness against human targets is mostly unchanged thanks to ecm, mcm, and sea skimming. The scimitar's performance is mostly unchanged because of its drag nerfs, making lofting rather useless.

You can use sim/grid_dymos.py to run the optimiser, it will make loft_tables.json for you. The process is something like this, I think? replace --jobs 15 with around twice the number of your cpu cores.

you will also need to run parse_unity.py to get a copy of coeffs_aam2.json.

```bash
uv venv
source .venv/bin/activate
uv pip install -r sim/requirements.txt
cd sim

python3 parse_unity.py --prefab ../txt/AAM2.txt --assets ../txt/GameAssets.txt --out coeffs_aam2.json

OMP_NUM_THREADS=1 python3 grid_dymos.py --grid grids/scythe2.json --jobs 15

python3 grid_dymos.py --grid grids/scythe2.json --merge
```

and then you can copy the table to mod/Tables. or the Tables directory next to the dll. I have provided some loft tables I have generated previously. I recommend just using those as they are probably good enough.

based on <https://doi.org/10.82124/CEAS-GNC-2026-016> but vibecoded so... again... the implementation is probably not great.
