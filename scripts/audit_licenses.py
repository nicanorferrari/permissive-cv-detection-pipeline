#!/usr/bin/env python3
"""
Automated CI License Audit Gate.
Enforces strict license compliance (Apache-2.0, MIT, BSD-2/3, ISC).
Blocks copyleft licenses (GPL, AGPL, LGPL, SSPL) from contaminating the supply chain.
"""

import sys
import subprocess
import json

PERMISSIVE_ALLOWLIST = {
    "apache-2.0",
    "apache 2.0",
    "apache software license",
    "mit",
    "mit license",
    "bsd-3-clause",
    "bsd license",
    "bsd-2-clause",
    "isc",
    "isc license",
    "python software foundation license",
    "psf",
    "unlicense",
    "cc0-1.0",
}

BLOCKED_PATTERNS = ["agpl", "gpl", "commercial", "proprietary"]

def run_license_audit():
    print("[+] Running dependency license audit...")
    try:
        # Run pip-licenses to inspect installed distribution licenses
        res = subprocess.run(
            ["pip-licenses", "--format=json", "--from=mixed"],
            capture_output=True,
            text=True,
            check=True
        )
        packages = json.loads(res.stdout)
    except Exception as e:
        print(f"[!] Warning: pip-licenses failed or not installed ({e}). Running fallback inspection...")
        # Basic fallback using importlib.metadata
        from importlib.metadata import distributions
        packages = []
        for dist in distributions():
            name = dist.metadata["Name"]
            license_val = dist.metadata.get("License", "UNKNOWN")
            packages.append({"Name": name, "License": license_val})

    violations = []
    checked_count = len(packages)

    for pkg in packages:
        name = pkg.get("Name", "")
        lic = pkg.get("License", "UNKNOWN").strip().lower()

        # Check for blocked copyleft signatures
        is_blocked = any(b in lic for b in BLOCKED_PATTERNS)
        is_allowed = any(a in lic for a in PERMISSIVE_ALLOWLIST)

        if is_blocked or not is_allowed:
            # Check if it's explicitly approved in allowlist
            if not is_allowed:
                violations.append((name, pkg.get("License", "UNKNOWN")))

    print(f"[+] Scanned {checked_count} packages.")

    if violations:
        print("\n" + "=" * 60)
        print("❌ LICENSE AUDIT FAILURE: NON-PERMISSIVE DEPENDENCIES FOUND")
        print("=" * 60)
        for name, lic in violations:
            print(f"  - Package: {name:<30} | License: {lic}")
        print("\nPolicy violation: Only MIT, Apache-2.0, BSD, and ISC licenses are allowed.")
        print("Ultralytics, AGPL-3.0 Albumentations, and copyleft libraries are strictly prohibited.")
        sys.exit(1)

    print("✅ All dependencies comply with Apache-2.0 / MIT / BSD permissive licensing policy.")
    sys.exit(0)

if __name__ == "__main__":
    run_license_audit()
