#!/usr/bin/env python3
"""Render-tier load test for the Typortrait engine.

Fires concurrent POST /render requests at a running container and reports, per
concurrency level: throughput (renders/min), latency percentiles, HTTP status
breakdown, and the PEAK queue depth read from /health. Use it to find the real
ceiling of ONE box BEFORE a traffic event (e.g. a GMA "Deals & Steals" spike).

Stdlib only -- runs on any host with python3, no pip installs.

  # Point at STAGING (port 8078) so a test never competes with live prod:
  python3 tools/stress_render.py --url http://127.0.0.1:8078 \
      --image /path/to/a_real_face.jpg --levels 1,2,4,8,16 --per-level 12

NOTE: rendering is CPU-bound and shares the box's cores. If you test a container
that is co-located with prod on the SAME VPS, the test WILL slow prod down while
it runs. Prefer an off-peak window, or a separate throwaway box of the same size.

The image MUST contain a real, front-facing face (the memorial engine needs
landmarks; a blank image returns 400 'needs_face' and measures nothing useful).
"""
import argparse
import json
import os
import time
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor


def build_multipart(fields, filefield, filename, filedata, filetype="image/jpeg"):
    boundary = "----typoStress%d" % int(time.time() * 1000)
    crlf = "\r\n"
    out = []
    for k, v in fields.items():
        out.append(("--" + boundary + crlf).encode())
        out.append(('Content-Disposition: form-data; name="%s"%s%s' % (k, crlf, crlf)).encode())
        out.append((str(v) + crlf).encode())
    out.append(("--" + boundary + crlf).encode())
    out.append(('Content-Disposition: form-data; name="%s"; filename="%s"%s'
                % (filefield, filename, crlf)).encode())
    out.append(("Content-Type: %s%s%s" % (filetype, crlf, crlf)).encode())
    out.append(filedata)
    out.append(crlf.encode())
    out.append(("--" + boundary + "--" + crlf).encode())
    return b"".join(out), boundary


def one_request(url, img_bytes, img_name, fields, timeout):
    body, boundary = build_multipart(fields, "image", img_name, img_bytes)
    req = urllib.request.Request(url + "/render", data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        r.read()
        return r.getcode(), time.time() - t0, None
    except urllib.error.HTTPError as e:
        try:
            detail = e.read()[:180].decode("utf-8", "replace")
        except Exception:
            detail = ""
        return e.code, time.time() - t0, detail
    except Exception as e:  # noqa: BLE001  (timeouts, resets, refused)
        return 0, time.time() - t0, str(e)[:180]


def get_health(url):
    try:
        r = urllib.request.urlopen(url + "/health", timeout=5)
        return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:120]}


def pctl(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


class HealthSampler(threading.Thread):
    """Poll /health in the background; remember the worst (deepest) queue seen."""
    def __init__(self, url, interval=0.5):
        super().__init__(daemon=True)
        self.url, self.interval = url, interval
        self.max_q = self.max_inflight = 0
        self.limit = None
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            h = get_health(self.url)
            r = (h or {}).get("render", h) if isinstance(h, dict) else {}
            q = (r or {}).get("queued", 0) or 0
            inf = (r or {}).get("in_flight", 0) or 0
            lim = (r or {}).get("concurrency_limit")
            if lim is not None:
                self.limit = lim
            self.max_q = max(self.max_q, q)
            self.max_inflight = max(self.max_inflight, inf)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def run_level(url, img_bytes, img_name, fields, concurrency, total, timeout):
    sampler = HealthSampler(url)
    sampler.start()
    lat, codes, errs = [], {}, []
    t0 = time.time()

    def task(_):
        return one_request(url, img_bytes, img_name, fields, timeout)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for code, dt, err in ex.map(task, range(total)):
            lat.append(dt)
            codes[code] = codes.get(code, 0) + 1
            if err and code != 200:
                errs.append((code, err))
    wall = time.time() - t0
    sampler.stop()
    return {
        "concurrency": concurrency, "total": total, "wall": wall,
        "rpm": (total / wall * 60.0) if wall > 0 else 0.0,
        "ok": codes.get(200, 0), "codes": codes,
        "p50": pctl(lat, 50), "p95": pctl(lat, 95), "max": max(lat) if lat else 0,
        "max_q": sampler.max_q, "max_inflight": sampler.max_inflight, "limit": sampler.limit,
        "errs": errs[:3],
    }


def main():
    ap = argparse.ArgumentParser(description="Render-tier load test")
    ap.add_argument("--url", default="http://127.0.0.1:8078", help="base URL (default: staging 8078)")
    ap.add_argument("--image", required=True, help="path to a REAL front-facing face photo")
    ap.add_argument("--levels", default="1,2,4,8", help="comma list of concurrency levels")
    ap.add_argument("--per-level", type=int, default=12, help="requests per level")
    ap.add_argument("--brand", default="lovedinwords")
    ap.add_argument("--style", default="displacement", help="displacement|words|message")
    ap.add_argument("--ground", default="navy")
    ap.add_argument("--backdrop", default="", help="floral key to also test the mat path (e.g. eucalyptus)")
    ap.add_argument("--png-width", type=int, default=1000, help="1000 ~ first preview (SS=1); 2000 ~ heavy")
    ap.add_argument("--words", default="beloved father grandfather kind gentle devoted faithful strong")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        img_bytes = f.read()
    img_name = os.path.basename(args.image)
    fields = {
        "words": args.words, "style": args.style, "ground": args.ground,
        "uppercase": "true", "png_width": str(args.png_width), "aspect": "0.8",
        "brand": args.brand, "biometric_consent": "1",
    }
    if args.backdrop:
        fields["backdrop"] = args.backdrop

    base = get_health(args.url)
    print("Target : %s" % args.url)
    print("Image  : %s (%d KB)" % (img_name, len(img_bytes) // 1024))
    print("Params : brand=%s style=%s ground=%s backdrop=%s png_width=%d"
          % (args.brand, args.style, args.ground, args.backdrop or "-", args.png_width))
    print("Health : %s" % json.dumps(base.get("render", base) if isinstance(base, dict) else base))
    print()
    hdr = "%-5s %-7s %-9s %-9s %-9s %-8s %-7s %-7s %s" % (
        "conc", "rpm", "p50(s)", "p95(s)", "max(s)", "ok/tot", "maxQ", "inflt", "codes/errs")
    print(hdr)
    print("-" * len(hdr))

    for lvl in [int(x) for x in args.levels.split(",") if x.strip()]:
        r = run_level(args.url, img_bytes, img_name, fields, lvl, args.per_level, args.timeout)
        codes = ",".join("%s:%d" % (k, v) for k, v in sorted(r["codes"].items()))
        note = codes
        if r["errs"]:
            note += " | " + "; ".join("%s %s" % (c, e) for c, e in r["errs"])
        print("%-5d %-7.1f %-9.2f %-9.2f %-9.2f %-8s %-7d %-7d %s" % (
            r["concurrency"], r["rpm"], r["p50"], r["p95"], r["max"],
            "%d/%d" % (r["ok"], r["total"]), r["max_q"], r["max_inflight"], note))

    print()
    print("Reading it:")
    print("  * rpm plateaus = your renders/min ceiling for ONE box. Divide a spike by it.")
    print("  * p95 climbing while rpm is flat = requests are QUEUING, not scaling.")
    print("  * non-200 codes or timeouts (code 0) = the box is shedding/dying under that load.")
    print("  * watch memory in another shell:  watch -n1 'free -h; docker stats --no-stream'")


if __name__ == "__main__":
    main()
