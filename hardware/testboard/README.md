# PS5 inspection test board v1

50 × 50 mm, 2-layer, 1.6 mm FR-4. Designed to produce every defect type in `kb/acceptance.yaml`.
Board map for the pipeline (`kb/board_demo/pos.csv`, `pads.csv`) is generated from this file with
`python -m tools.kicad_to_csv hardware/testboard/ps5_testboard.kicad_pcb kb/board_demo`.

## Circuit
5 V in (J1) → SS14 reverse protection (D6) → AMS1117-3.3 (U1) → 3V3.
74HC595 (U2, TSSOP-16) drives LEDs D1–D4; output QE switches power LED D5 through 2N7002 (Q1).
24C02 EEPROM (U3, SOIC-8) on I²C header J2. Shift-register inputs on J3.

## Status
- Placement: done (all SMD on top, pin-1 and polarity marks, 3 asymmetric fiducials, 4 × M3 holes)
- Nets: assigned on every pad
- GND: copper pour on both layers (all GND pads connect automatically)
- Routing: **to do** (about 20 connections)

## Finish in KiCad 10
1. Open `ps5_testboard.kicad_pcb` in the PCB Editor (accept the file-format upgrade).
2. Route the remaining ratsnest lines:
   - `+5V_IN`, `+5V`, `3V3`: 0.5 mm tracks
   - all other nets: 0.25 mm tracks
   - use vias to the bottom layer where tracks would cross
3. Press **B** to refill zones, then **Inspect → Design Rules Checker**: 0 errors, 0 unconnected.
4. Check before ordering: U1/U2/U3/Q1 pin 1, D1–D6 cathode marks, footprint sizes against the parts you buy.
5. **File → Fabrication Outputs**: Gerbers + drill files, and Component Placement (CPL) + BOM for assembly.

After routing, re-run the converter so the board map matches the final file.
