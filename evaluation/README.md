## Diverse Metrics Based on All Results

Note that for specific models, we need process the output prediction files.

### Best Performance Metric
```
python evaluation/calculate_best_acc.py
```

### Mean Relative Gain Metric
```
python evaluation/calculate_mean_relative_gain.py
```

### Stabability Metric
```
python evaluation/calculate_stabability.py
```

### Adaptability Metric
```
python evaluation/calculate_topk_hit_ratio.py
```