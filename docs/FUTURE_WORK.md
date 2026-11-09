# A testable next study

The current pilot motivates a more rigorous study; it does not establish that the question is novel or that focus improves decisions.

1. **Define the task and obtain appropriate data.** With a supervisor, select infection/ischaemia or wound-type labels and confirm data terms. Request DFUC2021 or another suitable wound dataset. Record availability of patient, acquisition-site and device identifiers before creating splits.
2. **Separate patients and sources.** Group all images from the same patient before fitting any model. If multiple acquisition sources are available, hold one source out. Do not claim source generalisation if the necessary metadata are absent.
3. **Validate the measurement.** Compare multiple focus measures with independently assessed image usability where permitted. Analyse effects of resolution, texture and acquisition. Treat controlled synthetic blur as a sensitivity check alongside naturally acquired images.
4. **Specify the comparison in advance.** Fix one primary coverage or risk target, selector baselines, calibration, class-wise metrics, uncertainty method and stopping rules before looking at the final test outcomes. Fit calibration and selection thresholds without test labels or test-score quantiles.
5. **Report null results and burden.** Publish class-specific error and deferral rates with intervals, plus the number of cases sent for review. A finding that focus adds no measurable value is still useful if the study has adequate data to distinguish meaningful effects.
6. **Assess clinical fit separately.** Any work with clinicians, new data collection or clinical use requires relevant institutional partnerships and approvals. Model metrics alone do not establish clinical usefulness.
