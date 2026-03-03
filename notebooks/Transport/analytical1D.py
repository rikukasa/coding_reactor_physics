import numpy as np
    
def analytical_sol_1D(SigT, Qm, dx, mu, w, div = 10, mid = False):
    NX = len(dx)
    NM = len(mu)
    
    # 任意の細分化数 n を設定
    n = div # 各区間を3分割する例

    # dx_detail を生成
    dx_detail = []
    for i in range(NX):
        dx_detail.extend([dx[i] / n] * n)  # 区間内を n 分割
    

    # SigT_detail を生成
    SigT_detail = []
    Qm_detail = []
    for i in range(NX):
        SigT_detail.extend([SigT[i]] * n)  # 各値を n 回繰り返す
        Qm_detail.extend([Qm[i]] * n)

    # リストを NumPy 配列に変換
    dx_detail = np.array(dx_detail)
    SigT_detail = np.array(SigT_detail)
    Qm_detail = np.array(Qm_detail)
    
    NNOD = len(dx_detail) + 1
    phi = np.zeros(NNOD)
    psi = np.zeros((NM, NNOD))
    for m in range(NM):
        if mu[m] > 0.0:
            startX = 1
            endX  = NNOD
            signX = 1
            delta_mat = -1
        else:
            startX = NNOD - 2
            endX  = -1
            signX = -1
            delta_mat = 0
        
        ds = dx_detail / abs(mu[m])
        for i in range(startX, endX, signX):
            phi_in = psi[m, i-signX]
            imat = i + delta_mat
            if SigT_detail[imat] != 0.0:
                psi[m, i] = phi_in * np.exp(-SigT_detail[imat] * ds[imat]) + Qm_detail[imat] / SigT_detail[imat] * (1 - np.exp(-SigT_detail[imat] * ds[imat]))
            else:
                psi[m, i] = phi_in + Qm_detail[imat] * ds[imat]
        
        phi += psi[m] * w[m]
    
    # 累積和を計算し、最初にゼロを追加
    dx_cumsum = np.insert(np.cumsum(dx_detail), 0, 0)
    
    return phi, psi, dx_cumsum