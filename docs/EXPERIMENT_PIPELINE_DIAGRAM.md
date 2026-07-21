# Experiment Pipeline — Input to Output

Sơ đồ tách toàn bộ thí nghiệm 7-arm × 150-episode thành bốn tầng. Sức mạnh
của thiết kế nằm ở chỗ: **một bộ đầu vào đông kết duy nhất** chảy qua **bảy
cấu hình controller khác nhau**, nên mọi khác biệt ở đầu ra quy được về đúng
một nguyên nhân — cấu hình arm.

```mermaid
flowchart TD
    subgraph S0["1 — Đầu vào đông kết (Stage 0, làm một lần)"]
        PAPER["Paper PDF đã xác minh<br/>eq. 18, Tables 1-2, protocol V-C"]
        PLAN["Plan JSON v1<br/>3 họ scenario × ranges × gates"]
        LHC["Latin Hypercube generator<br/>seed 20260716, stride 10000"]
        GATES["Preflight gates<br/>clearance đầu ≥ 0.03 m · encounter ≤ +0.06 m<br/>workspace box · cân bằng 25/25 · ≤1000 attempts"]
        INST["150 frozen instances<br/>sha256 e0f15460…"]
        PLAN --> LHC --> GATES --> INST
        PAPER -. "deviation registry" .-> PLAN
    end

    NIM["NVIDIA NIM llama-3.1-8b<br/>prompt hiệu chuẩn Table 1<br/>1 request thật → γ = 0.05<br/>checkpoint replay, 0 request thêm"]

    subgraph ARMS["2 — Bảy arm (mỗi arm một manifest ghim hash instance)"]
        M1["fixed static γ=0.15<br/>(baseline trung thực paper)"]
        M2["scripted feedback<br/>γ 0.07→0.03 theo mốc cứng"]
        M3["dead-time margin<br/>+0.21 m/s × tuổi đo"]
        M4["velocity tube<br/>lan truyền vận tốc + tube"]
        M5["soft slack<br/>s ≥ 0, phạt L1 w=1000"]
        M6["NIM feedback<br/>trên nền tube (cứng)"]
        M7["NIM + soft slack<br/>(đóng chuỗi nhân quả)"]
    end
    NIM --> M6
    NIM --> M7

    INST ==> RUNNER
    ARMS ==> CFG

    subgraph RUNNER["3 — Vòng điều khiển mỗi episode (≤260 bước, dt = 0.04 s)"]
        CFG["SmoothDynamicConfig<br/>hợp đồng đông kết + override của arm"]
        ENV["PandaReachSafe-v3<br/>PyBullet · Franka Emika Panda"]
        SENSOR["Sensor ZOH 0.67 s<br/>+ noise Gauss σ theo từng tập"]
        PRED["Obstacle prediction<br/>static ⟂ velocity-tube (± margin)"]
        TVP["TVP theo stage<br/>vị trí dự đoán + robust radius + γ(t)"]
        MPC["do-mpc / CasADi<br/>8-state · CBF: h_next ≥ (1−γ)·h<br/>(± slack trong ràng buộc và cost)"]
        IPOPT["IPOPT ≤ 0.035 s CPU"]
        SCREEN{"Acceptance<br/>screen"}
        ACT["action → env.step"]
        ZERO["fail-closed<br/>lệnh zero"]
        FB["gamma_schedule tại t_fb<br/>(scripted / NIM, TTL 10.4 s)"]
        CFG --> ENV --> SENSOR --> PRED --> TVP --> MPC --> IPOPT --> SCREEN
        SCREEN -- "OK" --> ACT --> ENV
        SCREEN -- "reject" --> ZERO --> ENV
        FB --> TVP
    end

    subgraph OUT["4 — Đầu ra và thống kê"]
        ROW["episodes.csv<br/>+ run_checkpoint.json mỗi row"]
        FAM["Theo họ: Wilson 95%<br/>bootstrap clearance 10k"]
        PAIR["Paired trên cùng instance:<br/>McNemar exact + Holm 3 họ"]
        SUM["benchmark_summary.json<br/>paired_summary.json"]
        DOC["RESULT doc — bảng 7 arm<br/>IEEE discussion doc"]
        ROW --> FAM --> SUM
        ROW --> PAIR --> SUM --> DOC
    end

    ENV -- "metrics mỗi bước: clearance thật/đo,<br/>CBF residual, solve time, failure/rejection" --> ROW
```

## Cách đọc theo dòng chảy nhân quả

| Tầng | Vai trò | Bất biến giữ chặt |
|---|---|---|
| 1 — Input | Sinh và đóng băng điều kiện thí nghiệm | Cùng 150 instance, cùng seed, hash khớp mọi nơi |
| 2 — Arms | Biến độc lập duy nhất của thí nghiệm | Manifest không được đụng khóa hợp đồng đông kết |
| 3 — Runner | Cỗ máy sinh dữ liệu, giống hệt cho mọi arm | Chỉ nhận khác biệt qua config; barrier không đổi |
| 4 — Output | Biến phụ thuộc + suy luận | Không gộp họ; paired trên đúng cặp episode |

## Ba "công tắc" tạo nên phát hiện chính (đều nằm ở tầng 3)

1. **PRED** static ⟂ velocity-tube — đòn bẩy an toàn lớn nhất (16% → 46%).
2. **FB** — tín hiệu siết γ, vô hại hay có hại tùy công tắc thứ ba.
3. **SCREEN → ZERO ⟂ MPC + slack** — chính nhánh "reject → zero" là cơ chế
   freeze biến feedback thành 77 va chạm; mở van slack (nhánh MPC hấp thụ
   violation) thì cùng tín hiệu FB cho 0 va chạm. Toàn bộ kết luận trung tâm
   của bài nằm ở việc bật/tắt đúng một cạnh này của sơ đồ.
