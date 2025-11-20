import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def solve_damped_spring_motion(t,f_data,initial_state):
    # ==========================================
    # 1. パラメータ設定
    # ==========================================
    m = 0.1       # 質量 [kg]
    k = 90.0      # ばね定数 [N/m]
    c = 1.8       # ★追加: 粘性減衰係数 [N*s/m] (値を大きくすると早く止まります)
    
    omega0_sq = k / m  # k/m
    damping_term = c / m # c/m

    # 補間関数の作成
    f_interpolated = interp1d(t, f_data, kind='linear', fill_value="extrapolate")

    # ==========================================
    # 3. 運動方程式の定義 (減衰あり)
    # ==========================================
    def model(state, t):
        u, v = state # u: 変位, v: 速度
        forcing = f_interpolated(t)
        
        dudt = v
        # ★変更: 減衰項 (- damping_term * v) を追加
        # dv/dt = (k/m)*(f - u) - (c/m)*v
        dvdt = omega0_sq * (forcing - u) - damping_term * v
        
        return [dudt, dvdt]

    # ==========================================
    # 4. 数値計算の実行
    # ==========================================

    solution = odeint(model, initial_state, t)
    u_t = solution[:, 0]
    return u_t

def add_dumping(x,maxdiff=20):
    t = np.arange(len(x))*0.005
    initial_state = [0.0, 0.0]
    u_t = solve_damped_spring_motion(t,x,initial_state)
    for i in range(len(u_t)):
        if x[i] == 0:
            u_t[i] = 0
        else:
            d = abs(u_t[i]-x[i])
            if d > maxdiff:
                u_t[i] = x[i]+(u_t[i]-x[i])/d*maxdiff
    return u_t

if __name__ == "__main__":
    import matplotlib.pyplot as plt

        # 時間設定 (0秒から20秒まで)
    t = np.linspace(0, 20, 1000)

    # ==========================================
    # 2. 入力信号 f(t) の作成
    # ==========================================
    # 今回は挙動が見やすいよう「ステップ入力」にします
    # (1秒後に支点を 1.0m 持ち上げて、そのまま固定)
    f_data = np.zeros_like(t)
    f_data[0:100] = 50
    f_data[100:200] = 100
    f_data[200:300] = 150
    f_data[300:400] = 100
    f_data[400:500] = 0
    f_data[500:600] = 100
    f_data[600:700] = 110
    f_data[700:800] = 80
    u_t = add_dumping(f_data)
    # ==========================================
    # 5. 結果の可視化
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 支点の動き
    plt.plot(t, f_data, 'r--', label='Support (Input) $f(t)$', alpha=0.6)
    
    # おもりの動き
    plt.plot(t, u_t, 'b-', label='Mass (Output) $u(t)$', linewidth=2)
    
    plt.title(f'Damped Vibration')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
