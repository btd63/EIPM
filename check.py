import numpy as np
import matplotlib.pyplot as plt

# ====== 경로 지정 (baseline 하나 선택) ======
npz_path = "./datasets/sim_linear_dx5_ntr250_nte10000_rpt100_tk5_ok5_seed42.npz"
# ↑ 실제 파일명에 맞게 수정

# ====== 저장 경로 ======
out_png = "./datasets/hist_logT_test_baseline.png"

# ====== 데이터 로드 ======
data = np.load(npz_path)

# log(T_test) 꺼내기 (필요하면 강제 로드)
logT_test = np.array(data["logT_test"])

# ====== Histogram ======
plt.figure()
plt.hist(logT_test[0], bins=50, density=True)
plt.xlabel("log(T_test)")
plt.ylabel("Density")
plt.title("Histogram of log(T_test) (baseline setting)")

# ====== 저장 ======
plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {out_png}")
