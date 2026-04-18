#!/usr/bin/env python3
"""Download clean music from YouTube: Hamilton, Ragtime, Wicked instrumentals,
Summer Salt, Bruno Mars, VGM, and Pokemon OSTs.

Uses yt-dlp to search and download. Targets instrumental/OST content to minimize
vocal contamination (Demucs strip_vocals.py handles the rest).
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path

BREW_BIN = "/opt/homebrew/bin"
ENV = {**os.environ, "PATH": f"{BREW_BIN}:{os.environ.get('PATH', '')}"}

# --- Hamilton (instrumental/karaoke versions to reduce vocal stripping needed) ---
HAMILTON_QUERIES = [
    "ytsearch5:Hamilton musical instrumental",
    "ytsearch5:Hamilton Broadway karaoke instrumental",
    "ytsearch5:Hamilton original soundtrack instrumental",
    "ytsearch5:Hamilton the musical orchestral backing track",
    "ytsearch3:Hamilton act 1 instrumental",
    "ytsearch3:Hamilton act 2 instrumental",
    # Individual well-known songs (instrumental versions)
    "ytsearch3:My Shot Hamilton instrumental",
    "ytsearch3:Alexander Hamilton instrumental backing track",
    "ytsearch3:Wait For It Hamilton instrumental",
    "ytsearch3:Satisfied Hamilton instrumental",
    "ytsearch3:The Room Where It Happens Hamilton instrumental",
    "ytsearch3:Non-Stop Hamilton instrumental",
    "ytsearch3:Yorktown Hamilton instrumental",
    "ytsearch3:Burn Hamilton instrumental",
    "ytsearch3:It's Quiet Uptown Hamilton instrumental",
    "ytsearch3:Who Lives Who Dies Who Tells Your Story Hamilton instrumental",
]

# --- Ragtime Musical ---
RAGTIME_QUERIES = [
    "ytsearch5:Ragtime musical instrumental",
    "ytsearch5:Ragtime Broadway soundtrack instrumental",
    "ytsearch3:Ragtime musical karaoke backing track",
    "ytsearch3:Ragtime the musical orchestral",
    "ytsearch3:Wheels of a Dream Ragtime instrumental",
    "ytsearch3:New Music Ragtime musical instrumental",
    "ytsearch3:Back to Before Ragtime instrumental",
    "ytsearch3:Make Them Hear You Ragtime instrumental",
    "ytsearch3:Coalhouse's Soliloquy Ragtime instrumental",
]

# --- Wicked Musical ---
WICKED_QUERIES = [
    "ytsearch5:Wicked musical instrumental",
    "ytsearch5:Wicked Broadway soundtrack instrumental",
    "ytsearch5:Wicked karaoke instrumental backing track",
    "ytsearch3:Defying Gravity instrumental",
    "ytsearch3:Popular Wicked instrumental",
    "ytsearch3:For Good Wicked instrumental",
    "ytsearch3:No Good Deed Wicked instrumental",
    "ytsearch3:What Is This Feeling Wicked instrumental",
    "ytsearch3:The Wizard and I instrumental",
    "ytsearch3:Dancing Through Life Wicked instrumental",
    "ytsearch3:As Long As You're Mine Wicked instrumental",
]

# --- Summer Salt ---
SUMMER_SALT_QUERIES = [
    "ytsearch10:Summer Salt full album",
    "ytsearch5:Summer Salt Driving to Hawaii",
    "ytsearch5:Summer Salt Happy Camper full album",
    "ytsearch5:Summer Salt Sequoia Moon full album",
    "ytsearch5:Summer Salt Going Native full album",
    "ytsearch3:Summer Salt Revvin' My CJ7",
    "ytsearch3:Summer Salt Sweet to Me",
    "ytsearch3:Summer Salt Rockaway",
]

# --- Bruno Mars ---
BRUNO_MARS_QUERIES = [
    "ytsearch5:Bruno Mars instrumental",
    "ytsearch5:Bruno Mars karaoke instrumental",
    "ytsearch3:Just The Way You Are Bruno Mars instrumental",
    "ytsearch3:Treasure Bruno Mars instrumental",
    "ytsearch3:24K Magic Bruno Mars instrumental",
    "ytsearch3:Locked Out of Heaven Bruno Mars instrumental",
    "ytsearch3:That's What I Like Bruno Mars instrumental",
    "ytsearch3:When I Was Your Man Bruno Mars instrumental",
    "ytsearch3:Uptown Funk Bruno Mars instrumental",
    "ytsearch3:Leave The Door Open Bruno Mars instrumental",
    "ytsearch3:Finesse Bruno Mars instrumental",
    "ytsearch3:Grenade Bruno Mars instrumental",
]

# --- Video Game Music (instrumental by nature) ---
VGM_QUERIES = [
    "ytsearch20:video game music OST full album",
    "ytsearch15:video game soundtrack instrumental",
    "ytsearch10:JRPG soundtrack OST",
    "ytsearch10:Nintendo soundtrack OST",
    # Specific well-known OSTs
    "ytsearch5:Zelda Breath of the Wild OST",
    "ytsearch5:Zelda Tears of the Kingdom OST",
    "ytsearch5:Zelda Ocarina of Time OST",
    "ytsearch5:Final Fantasy VII OST",
    "ytsearch5:Final Fantasy X OST",
    "ytsearch5:Hollow Knight OST full",
    "ytsearch5:Celeste OST full",
    "ytsearch5:Undertale OST full",
    "ytsearch5:Stardew Valley OST",
    "ytsearch5:Minecraft soundtrack volume alpha",
    "ytsearch5:Animal Crossing New Horizons OST",
    "ytsearch5:Super Mario Galaxy OST",
    "ytsearch5:Chrono Trigger OST",
    "ytsearch5:Persona 5 OST instrumental",
    "ytsearch5:Nier Automata OST",
    "ytsearch5:Ori and the Blind Forest OST",
    "ytsearch5:Journey game OST",
    "ytsearch5:Hades game OST",
    "ytsearch5:Cuphead OST",
    "ytsearch5:Dark Souls OST",
    "ytsearch5:Skyrim OST ambient",
    "ytsearch5:Fire Emblem Three Houses OST",
    "ytsearch5:Kirby OST",
    "ytsearch5:Mega Man OST",
    "ytsearch5:Metroid Prime OST",
    "ytsearch5:Splatoon OST",
]

# --- Pokemon OSTs (all mainline games) ---
POKEMON_QUERIES = [
    # Gen 1
    "ytsearch5:Pokemon Red Blue OST full",
    "ytsearch5:Pokemon Yellow OST full",
    # Gen 2
    "ytsearch5:Pokemon Gold Silver OST full",
    "ytsearch5:Pokemon Crystal OST full",
    # Gen 3
    "ytsearch5:Pokemon Ruby Sapphire OST full",
    "ytsearch5:Pokemon Emerald OST full",
    "ytsearch5:Pokemon FireRed LeafGreen OST full",
    # Gen 4
    "ytsearch5:Pokemon Diamond Pearl OST full",
    "ytsearch5:Pokemon Platinum OST full",
    "ytsearch5:Pokemon HeartGold SoulSilver OST full",
    # Gen 5
    "ytsearch5:Pokemon Black White OST full",
    "ytsearch5:Pokemon Black 2 White 2 OST full",
    # Gen 6
    "ytsearch5:Pokemon X Y OST full",
    "ytsearch5:Pokemon Omega Ruby Alpha Sapphire OST full",
    # Gen 7
    "ytsearch5:Pokemon Sun Moon OST full",
    "ytsearch5:Pokemon Ultra Sun Moon OST full",
    # Gen 8
    "ytsearch5:Pokemon Sword Shield OST full",
    "ytsearch5:Pokemon Brilliant Diamond Shining Pearl OST full",
    "ytsearch5:Pokemon Legends Arceus OST full",
    # Gen 9
    "ytsearch5:Pokemon Scarlet Violet OST full",
    # Spinoffs with notable soundtracks
    "ytsearch3:Pokemon Mystery Dungeon Explorers of Sky OST",
    "ytsearch3:Pokemon Colosseum OST",
    "ytsearch3:Pokemon Stadium OST",
    "ytsearch3:Pokemon Snap OST",
]

MAX_DURATION = 1200  # 20 min (some OST compilations are long single videos)
MIN_DURATION = 30


def download_query(query: str, out_dir: Path, archive_file: Path, dry_run: bool = False):
    cmd = [
        "/opt/homebrew/bin/yt-dlp",
        "--ffmpeg-location", "/opt/homebrew/bin/ffmpeg",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--cookies-from-browser", "firefox",
        "--output", str(out_dir / "%(title)s__%(id)s.%(ext)s"),
        "--match-filter", f"duration>{MIN_DURATION} & duration<{MAX_DURATION}",
        "--download-archive", str(archive_file),
        "--ignore-errors",
        "--no-overwrites",
        "--restrict-filenames",
    ]

    if query.startswith("http"):
        cmd.append("--yes-playlist")
    else:
        cmd.append("--no-playlist")

    if dry_run:
        cmd.append("--simulate")
    else:
        cmd.append("--print-json")

    cmd.append(query)

    if dry_run:
        print(f"  [DRY RUN] {query}")
        return []

    print(f"  Downloading: {query}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=ENV)

    downloaded = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            downloaded.append({
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "url": info.get("webpage_url"),
                "category": "unknown",
            })
        except json.JSONDecodeError:
            pass

    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().split("\n"):
            if "ERROR" in line:
                print(f"    Warning: {line.strip()}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Crawl YouTube for clean music (Hamilton, VGM, Pokemon)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--category", choices=["all", "hamilton", "ragtime", "wicked", "summer_salt", "bruno_mars", "vgm", "pokemon"], default="all")
    parser.add_argument("--extra-queries", nargs="*", default=[])
    args = parser.parse_args()

    base = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "data" / "raw" / "clean"
    base.mkdir(parents=True, exist_ok=True)
    archive_file = base / ".yt_archive.txt"
    manifest_file = base / "download_manifest.jsonl"

    queries = []
    if args.category in ("all", "hamilton"):
        queries += HAMILTON_QUERIES
    if args.category in ("all", "ragtime"):
        queries += RAGTIME_QUERIES
    if args.category in ("all", "wicked"):
        queries += WICKED_QUERIES
    if args.category in ("all", "summer_salt"):
        queries += SUMMER_SALT_QUERIES
    if args.category in ("all", "bruno_mars"):
        queries += BRUNO_MARS_QUERIES
    if args.category in ("all", "vgm"):
        queries += VGM_QUERIES
    if args.category in ("all", "pokemon"):
        queries += POKEMON_QUERIES
    queries += [f"ytsearch10:{q}" for q in args.extra_queries]

    print(f"Clean music crawler: {len(queries)} queries, output to {base}")
    if args.category != "all":
        print(f"  Category filter: {args.category}")

    total = 0
    with open(manifest_file, "a") as mf:
        for i, query in enumerate(queries):
            print(f"[{i+1}/{len(queries)}]", end="")
            try:
                results = download_query(query, base, archive_file, args.dry_run)
                for r in results:
                    q_lower = query.lower()
                    if "hamilton" in q_lower:
                        r["category"] = "hamilton"
                    elif "ragtime" in q_lower:
                        r["category"] = "ragtime"
                    elif "wicked" in q_lower:
                        r["category"] = "wicked"
                    elif "summer salt" in q_lower:
                        r["category"] = "summer_salt"
                    elif "bruno mars" in q_lower:
                        r["category"] = "bruno_mars"
                    elif "pokemon" in q_lower or "pokémon" in q_lower:
                        r["category"] = "pokemon"
                    else:
                        r["category"] = "vgm"
                    mf.write(json.dumps(r) + "\n")
                    total += 1
            except subprocess.TimeoutExpired:
                print(f"    Timeout on: {query}")
            except Exception as e:
                print(f"    Error on {query}: {e}")

    print(f"\nDone. Downloaded {total} tracks to {base}")
    print(f"Manifest: {manifest_file}")
    print(f"Archive (dedup): {archive_file}")


if __name__ == "__main__":
    main()
