# Dashboard test checklist (laptop or Pi)

Mark each line when it behaves as described. Write down anything that does not, with a screenshot.

## Start
- [ ] `python -m ui.fabric_app` starts; page opens at port 5000 and shows the fabric selector, Pattern check button, Live camera card.

## Upload and crop
- [ ] Inspect photo, choose a photo: page shows "Crop an area" and "Inspect full photo".
- [ ] Inspect full photo: result shows "uploaded photo, not straightened" and "Area inspected: whole photo".
- [ ] Crop an area, drag a rectangle, let go: the cropped part is shown straight away.
- [ ] Inspect this part: result shows the cropped size in "Area inspected".
- [ ] A very small rectangle is refused with a message. Cancel and Crop again work.
- [ ] A rectangle under half of the photo shows the "small area" note.

## Live camera
- [ ] Start live view: picture is not black; Stop camera releases the camera.
- [ ] Capture and select area, then inspect: "calibrated" scale shown (camera pictures are straightened).
- [ ] Camera in use by another program: a clear message, not an error page.

## Result page
- [ ] Verdict colour, why sentence, numbered boxes on the picture, recommended action (no "from rules" tag).
- [ ] Operator actions (Llama) show for a defect and are skipped for a normal photo.
- [ ] Correct / wrong / missed buttons save and show the confirmation.

## Log and report
- [ ] Recent inspections: click a time opens the report. Full log lists more.
- [ ] Report shows the picture, a summary for every defect, actions, timings and the operator confirmation.
- [ ] Print / save as PDF works. An old entry without a picture shows the note instead.

## Pattern check
- [ ] Add 8 or more clean reference photos; Check a clean photo (no gap); check `pat_test_good_01_miss1.png` (gap marked).
- [ ] Delete a reference photo; the count goes down.

## Fabric selector
- [ ] Switch blue / pattern; the result uses the chosen fabric.
