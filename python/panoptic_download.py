# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Download the Kinect data of CMU Panoptic sequences.

Python port of the Kinect part of the panoptic-toolbox script
`scripts/getData_kinoptic.sh` (https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox):
the same files from the same server, into the same layout, so the toolbox
script can be used instead. Interrupted downloads are resumed.

    <out_root>/<seq>/calibration_<seq>.json
    <out_root>/<seq>/kcalibration_<seq>.json
    <out_root>/<seq>/synctables_<seq>.json, ksynctables_<seq>.json
    <out_root>/<seq>/kinectVideos/kinect_50_KK.mp4        (~150 MB each)
    <out_root>/<seq>/kinect_shared_depth/KINECTNODE<K>/depthdata.dat (~2.6 GB each)

Example (the two sequences of the paper, ~28 GB each):

    python python/panoptic_download.py --out_root <dataset>/panoptic-toolbox 170407_office2 160906_band2

Please follow the dataset's terms of use (http://domedb.perception.cs.cmu.edu).
"""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINTS = {
    "cmu": "http://domedb.perception.cs.cmu.edu",
    "snu": "http://vcl.snu.ac.kr/panoptic",  # mirror (--snu-endpoint of the toolbox)
}


def download(url: str, dst: Path) -> None:
    """Download `url` to `dst`, resuming a partial file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if dst.exists():
        print(f"[skip] {dst}")
        return
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # the partial file is already complete
            part.replace(dst)
            return
        raise RuntimeError(f"{url}: HTTP {e.code}") from e
    with response:
        resumed = response.status == 206
        total = response.headers.get("Content-Length")
        print(f"[get ] {url} -> {dst}" + (f" ({int(total) / 2**20:.0f} MB)" if total else "")
              + (f", resuming at {offset / 2**20:.0f} MB" if resumed else ""))
        with part.open("ab" if resumed else "wb") as f:
            shutil.copyfileobj(response, f, length=1 << 22)
    part.replace(dst)


def download_sequence(seq: str, out_root: Path, endpoint: str, kinect_ids: list[int]) -> None:
    base = f"{endpoint}/webdata/dataset/{seq}"
    seq_dir = out_root / seq
    download(f"{base}/calibration_{seq}.json", seq_dir / f"calibration_{seq}.json")
    download(f"{base}/kinect_shared_depth/kcalibration_{seq}.json", seq_dir / f"kcalibration_{seq}.json")
    download(f"{base}/kinect_shared_depth/synctables.json", seq_dir / f"synctables_{seq}.json")
    download(f"{base}/kinect_shared_depth/ksynctables.json", seq_dir / f"ksynctables_{seq}.json")
    for k in kinect_ids:
        download(f"{base}/videos/kinect_shared_crf20/{seq}_kinect{k}.mp4",
                 seq_dir / "kinectVideos" / f"kinect_50_{k:02d}.mp4")
    for k in kinect_ids:
        name = f"kinect_shared_depth/KINECTNODE{k}/depthdata.dat"
        download(f"{base}/{name}", seq_dir / name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sequences", nargs="+", help="sequence names, e.g. 170407_office2 160906_band2")
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="cmu")
    p.add_argument("--num_kinects", type=int, default=10, help="Kinects 1..N (the paper uses all 10)")
    args = p.parse_args()
    for seq in args.sequences:
        download_sequence(seq, args.out_root, ENDPOINTS[args.endpoint],
                          list(range(1, args.num_kinects + 1)))


if __name__ == "__main__":
    main()
