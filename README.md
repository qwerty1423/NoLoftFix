# NOLoftFix

This project is an unofficial community modification and is not affiliated with, sponsored by, or endorsed by Shockfront Studios Pty Ltd. Original Nuclear Option assets, vehicle designs, audio, and code are Copyright (c) 2026 Shockfront Studios Pty Ltd. All rights reserved. Nuclear Option and Shockfront Studios are trademarks or registered trademarks of Shockfront Studios Pty Ltd. Original mod content and all other trademarks belong to their respective owners.

---
Vanilla 0.34.2
<img alt="lofting without loft fix" src="https://github.com/user-attachments/assets/745f1cd8-7ed5-488e-8fc6-a057aefedefc" />

With loft fix
<img alt="lofting with loft fix" src="https://github.com/user-attachments/assets/a2b648b9-ea11-4456-b546-1fdd122d115f" />

LLMs were used to make this mod

some files were omitted from the repo

code quality may be low

The mod only patches code that runs if you are host, so you can safely join vanilla games with it if you want, as it will have no effect. If you are the host, then the mod will change the lofting behaviour, and vanilla clients should still be able to join your game.

May improve scythe performance. The scythe will now hit targets up to around 250 km as long as they are easy targets such as AI helicopters, by lofting properly. AI fighters can be hit around half the time at around 90 km. Effectiveness against human targets is mostly unchanged thanks to ecm, mcm, and sea skimming.

I have provided some loft tables I have generated previously, I recommend just using them as they are probably good enough.

Part of the scythe table:
<img alt="scythe2 2 1t" src="https://github.com/user-attachments/assets/ea30ad30-c657-4ed5-a827-a4305116bfbb" />

Part of the scimitar table:
<img alt="scimitar2t" src="https://github.com/user-attachments/assets/1a9e5131-0755-4d5a-92d1-e4687ac9a9df" />

It seems like you shouldn't launch scimitars at low speed?

There are nonsensical trajectories in both tables but they usually get averaged out, and most are at longer ranges anyway.

If you want to optimise some missiles yourself, you can use sim/grid_dymos.py to run the optimiser, it will make loft_tables.json for you. After cloning the repo, the process is something like this, I think? replace --jobs 15 with around twice the number of your cpu cores.

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

and then you can copy the table to mod/Tables. or the Tables directory next to the dll.

based on <https://doi.org/10.82124/CEAS-GNC-2026-016> but vibecoded so... again... the implementation is probably not great.
