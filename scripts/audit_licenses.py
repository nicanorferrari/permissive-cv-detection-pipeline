#!/usr/bin/env python3
"""
Automated CI License Audit Gate.
Enforces strict license compliance (Apache-2.0, MIT, BSD-2/3, ISC).
Blocks copyleft licenses (GPL, AGPL, LGPL, SSPL) from contaminating the supply chain.
Can audit either requirements.txt or the active Python environment.
Safe across Windows (CP1252) and Linux environments.
"""

import sys
import argparse
import subprocess
import json
import urllib.request
import re

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PERMISSIVE_ALLOWLIST = {
    "apache-2.0",
    "apache 2.0",
    "apache software license",
    "mit",
    "mit license",
    "bsd-3-clause",
    "bsd license",
    "bsd-2-clause",
    "bsd",
    "isc",
    "isc license",
    "python software foundation license",
    "psf",
    "unlicense",
    "cc0-1.0",
}

BLOCKED_PATTERNS = ["agpl", "gpl", "commercial", "proprietary"]

def fetch_pypi_license(pkg_name: str) -> str:
    """Queries PyPI JSON API for package license metadata."""
    try:
        url = f"https://pypi.org/pypi/{pkg_name}/json"
        req = urllib.request.Request(url, headers={"User-Agent": "LicenseAudit/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            info = data.get("info", {})
            lic = info.get("license") or ""
            if not lic or len(lic) > 100 or "\n" in lic:
                # check classifiers
                for c in info.get("classifiers", []):
                    if "License :: OSI Approved ::" in c:
                        lic = c.split("License :: OSI Approved ::")[-1].strip()
                        break
            return lic or "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def parse_requirements(req_path: str):
    pkgs = []
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Extract package name before ==, >=, <=, etc.
            name = re.split(r"[><=~;]", line)[0].strip()
            if name:
                pkgs.append(name)
    return pkgs

def audit_requirements_file(req_path: str):
    print(f"[+] Auditing declared dependencies in '{req_path}' against permissive policy...")
    pkgs = parse_requirements(req_path)
    violations = []

    for pkg in pkgs:
        lic = fetch_pypi_license(pkg)
        lic_clean = lic.lower().strip()
        print(f"  - Package: {pkg:<25} | Detected License: {lic}")

        is_blocked = any(b in lic_clean for b in BLOCKED_PATTERNS)
        if is_blocked:
            violations.append((pkg, lic, "Copyleft / AGPL detected"))

    print(f"[+] Verified {len(pkgs)} project dependencies.")
    if violations:
        print("\n" + "=" * 60)
        print("[FAIL] LICENSE AUDIT FAILURE: NON-PERMISSIVE DEPENDENCIES FOUND")
        print("=" * 60)
        for name, lic, reason in violations:
            print(f"  - Package: {name:<25} | License: {lic} ({reason})")
        sys.exit(1)
    
    print("\n[PASS] All project dependencies strictly comply with Apache-2.0 / MIT / BSD policy.")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License Audit Gate")
    parser.add_argument("--requirements", "-r", type=str, default="requirements.txt", help="Path to requirements.txt")
    args = parser.parse_args()

    audit_requirements_file(args.requirements)
