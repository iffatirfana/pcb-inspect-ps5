# PS5 inspection test board v1

50 × 50 mm, 2-layer, 1.6 mm FR-4. Designed to produce every defect type in `kb/acceptance.yaml`.

## Circuit
5 V in (J1) → SS14 reverse protection (D6) → AMS1117-3.3 (U1) → 3V3.
74HC595 (U2, TSSOP-16) drives LEDs D1–D4; output QE switches power LED D5 through 2N7002 (Q1).
24C02 EEPROM (U3, SOIC-8) on I²C header J2. Shift-register inputs on J3.

## Status
| Item | State |
|---|---|
| Placement | Done – all SMD on top, pin-1 / polarity marks, 3 asymmetric fiducials, 4 × M3 holes |
| Routing | Done – autorouted (Freerouting), 0.5 mm power / 0.25 mm signal tracks |
| Ground | Pours on both layers + GND stitching vias |
| DRC | 0 clearance errors, 0 unconnected |

## Files
| File | Use |
|---|---|
| `ps5_testboard.kicad_pcb` | KiCad board (open in PCB Editor) |
| `fab/ps5_testboard_gerbers.zip` | Gerbers + drill – upload to the PCB fab |
| `fab/ps5_testboard_BOM.csv` | Bill of materials |
| `fab/ps5_testboard_CPL.csv` | Pick-and-place positions for SMT assembly (top side). J1–J3 are through-hole headers. |
| `ps5_testboard_routed.png` | Top (red) and bottom (blue) copper preview |

## Before ordering
1. Open the board in KiCad 10, press **B** (refill zones), run **Inspect → Design Rules Checker**.
2. Check pin 1 of U1/U2/U3/Q1 and the cathode marks of D1–D6 against the parts you buy.
3. Check the fab's assembly rotation conventions against `CPL.csv` in their online preview.

The board map used by the pipeline (`kb/board_demo/pos.csv`, `pads.csv`) is generated from this file:
`python -m tools.kicad_to_csv hardware/testboard/ps5_testboard.kicad_pcb kb/board_demo`
