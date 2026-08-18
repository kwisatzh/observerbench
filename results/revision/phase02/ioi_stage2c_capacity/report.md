# Stage 2c primary-stratified design: capacity-matched IOI pair analysis

Input: results/frozen/ioi/stage2c_primary_stratified_mean_end. The prompt bootstrap resamples prompts while holding the intervention masks and ten repeated five-fold splits fixed.

Each pair receives the same four-column count-bin interaction block.

## Add one, relative to count-additive

- PB: 0.00019 [-0.00106, 0.00141]
- PE: 0.12316 [0.11682, 0.12970]
- BE: -0.00038 [-0.00133, 0.00058]

## Add one, relative to additive-head

- PB: -0.00622 [-0.00799, -0.00458]
- PE: 0.11675 [0.11093, 0.12255]
- BE: -0.00679 [-0.00840, -0.00524]

## Leave one pair out of the all-pairs model

- PB: 0.00964 [0.00811, 0.01103]
- PE: 0.13294 [0.12571, 0.14053]
- BE: 0.00216 [0.00102, 0.00357]
