# cg_common.jl
using JLD2, LinearAlgebra, Statistics

# --- 投影関数 (射影変換 + レンズ歪み) ---
function project_point_fast(p_world, rvec, tvec, mtx, dist)
    theta = norm(rvec)
    R = (theta < 1e-10) ? Matrix{Float64}(I, 3, 3) : begin
        k = rvec / theta
        K_cross = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0]
        Matrix{Float64}(I, 3, 3) + sin(theta) * K_cross + (1 - cos(theta)) * (k * k' - I)
    end
    P_cam = R * p_world + tvec
    if P_cam[3] <= 0 return NaN, NaN end
    x, y = P_cam[1]/P_cam[3], P_cam[2]/P_cam[3]
    
    # 歪み補正
    r2 = x^2 + y^2
    k1,k2,p1,p2,k3 = dist[1],dist[2],dist[3],dist[4],dist[5]
    radial = 1.0 + k1*r2 + k2*r2^2 + k3*r2^3
    dx = 2*p1*x*y + p2*(r2 + 2*x^2)
    dy = p1*(r2 + 2*y^2) + 2*p2*x*y
    u = mtx[1,1]*(x*radial + dx) + mtx[1,3]
    v = mtx[2,2]*(y*radial + dy) + mtx[2,3]
    return u, v
end

# --- キャリブレーションデータ読み込み ---
function load_calib_data(path="calibration_data.jld2")
    if !isfile(path) error("$path が見つかりません") end
    d = load(path)
    # 辞書形式で返す
    return (
        mtx1=d["mtx1_jl"], dist1=d["dist1_jl"], r1=d["r1"], t1=d["t1"],
        mtx2=d["mtx2_jl"], dist2=d["dist2_jl"], r2=d["r2"], t2=d["t2"],
        pos1=d["pos1"], pos2=d["pos2"],
        board_w = d["NX"] * d["MARKER_LEN"] + (d["NX"]-1) * d["MARKER_SEP"],
        board_h = d["NY"] * d["MARKER_LEN"] + (d["NY"]-1) * d["MARKER_SEP"]
    )
end