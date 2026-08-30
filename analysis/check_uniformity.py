import pandas as pd, numpy as np, cv2, os
for t in ["generated_track","cond1_light","cond1_trees","cond1_cones"]:
    d = pd.read_csv(f"data_cte/{t}/cte_log.csv").sample(300, random_state=42)
    within, grads = [], []
    for p in d["image"]:
        img = cv2.imread(os.path.join(f"data_cte/{t}", p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        roi = img[80:120, :]                      # near-field road strip
        within.append(roi.std())                  # contrast WITHIN each frame
        grads.append(abs(float(roi[:,:40].mean()) - float(roi[:,120:].mean())))
    print(f"{t:16s} within-frame contrast {np.mean(within):5.1f}   "
          f"left-right asymmetry {np.mean(grads):5.1f}")
