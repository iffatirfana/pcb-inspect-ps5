"""Convert the KiCad BOM/CPL in hardware/testboard/fab/ to JLCPCB assembly format.

Run from the repo root:  python -m tools.jlc_export
Writes  fab/ps5_testboard_BOM_JLC.csv  and  fab/ps5_testboard_CPL_JLC.csv
Through-hole headers (J*) are left out: solder them by hand.
"""
import csv
from pathlib import Path

FAB = Path("hardware/testboard/fab")
SKIP_PREFIX = ("J",)  # not machine-assembled

# value + footprint keyword -> LCSC part number (checked on jlcpcb.com/partdetail/...)
LCSC = {
    ("10uF", "0805"): "C15850",        # Samsung CL21A106KAYNNNE        basic
    ("100nF", "0603"): "C14663",       # YAGEO CC0603KRX7R9BB104        basic
    ("100nF", "0402"): "C1525",        # Samsung CL05B104KO5NNNC        basic
    ("LED", "0805"): "C84256",         # red 0805                       basic
    ("PWR", "0805"): "C84256",         # same red LED                   basic
    ("SS14", "SMA"): "C2480",          # MDD SS14                       basic
    ("2N7002", "SOT-23"): "C8545",     # CJ 2N7002                      basic
    ("1k", "0603"): "C21190",          # UNI-ROYAL 0603WAF1001T5E       basic
    ("4.7k", "0603"): "C23162",        # UNI-ROYAL 0603WAF4701T5E       basic
    ("100R", "0402"): "C25076",        # UNI-ROYAL 0402WGF1000TCE       basic
    ("10k", "0402"): "C25744",         # UNI-ROYAL 0402WGF1002TCE       basic
    ("AMS1117-3.3", "SOT-223"): "C6186",  # AMS1117-3.3                 basic
    ("74HC595", "TSSOP-16"): "C5948",  # Nexperia 74HC595PW,118         extended
    ("24C02", "SOIC-8"): "C6203",      # Microchip AT24C02C-SSHM-T      extended
}


def lcsc_for(value, footprint):
    for (v, fp), part in LCSC.items():
        if v == value and fp in footprint:
            return part
    raise KeyError(f"no LCSC part for {value} / {footprint}")


def main():
    with open(FAB / "ps5_testboard_BOM.csv", newline="", encoding="utf-8") as f:
        bom = list(csv.DictReader(f))
    with open(FAB / "ps5_testboard_BOM_JLC.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Comment", "Designator", "Footprint", "LCSC Part #"])
        n = 0
        for r in bom:
            refs = [x for x in r["Designators"].split() if not x.startswith(SKIP_PREFIX)]
            if not refs:
                continue
            w.writerow([r["Value"], ",".join(refs), r["Footprint"],
                        lcsc_for(r["Value"], r["Footprint"])])
            n += len(refs)

    with open(FAB / "ps5_testboard_CPL.csv", newline="", encoding="utf-8") as f:
        cpl = list(csv.DictReader(f))
    with open(FAB / "ps5_testboard_CPL_JLC.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
        m = 0
        for r in cpl:
            if r["Ref"].startswith(SKIP_PREFIX):
                continue
            w.writerow([r["Ref"], f"{float(r['PosX']):.3f}mm", f"{float(r['PosY']):.3f}mm",
                        "Top" if r["Side"] == "top" else "Bottom", f"{float(r['Rot']):g}"])
            m += 1
    print(f"BOM: {n} placed parts | CPL: {m} placements | headers J* left for hand soldering")
    assert n == m, "BOM and CPL counts differ"


if __name__ == "__main__":
    main()
