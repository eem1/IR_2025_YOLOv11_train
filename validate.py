from ultralytics import YOLO

# Load weights
model = YOLO('runs/detect/run1/weights/best.pt')

# Run validation with verbose mode on
# 'plots=True' ensures confusion matrices and curves are saved
metrics = model.val(data='data.yaml', verbose=True, plots=True)

print("\n--- FULL METRIC BREAKDOWN ---")

# 1. The Standard Numbers (Aggregate)
print(f"Mean Precision: {metrics.results_dict['metrics/precision(B)']:.3f}")
print(f"Mean Recall:    {metrics.results_dict['metrics/recall(B)']:.3f}")
print(f"mAP50:          {metrics.results_dict['metrics/mAP50(B)']:.3f}")
print(f"mAP50-95:       {metrics.results_dict['metrics/mAP50-95(B)']:.3f}")

# 2. Per-Class Breakdown (Hidden in the 'maps' array)
# This shows the mAP50-95 for EVERY individual class ID
# Useful if you are detecting 'Person', 'Car', 'Deer' and want to know which one is failing.
print(f"\nmAP50-95 per class: {metrics.box.maps}") 

# 3. Speed Metrics (Hidden details)
# Crucial for thermal cameras if you need real-time performance
print(f"\nSpeed (ms):")
print(f"  Pre-process:  {metrics.speed['preprocess']:.2f}ms")
print(f"  Inference:    {metrics.speed['inference']:.2f}ms")
print(f"  Post-process: {metrics.speed['postprocess']:.2f}ms")