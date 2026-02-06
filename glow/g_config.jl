#g_config.jl
# ==========================================
# Part 1: 電極モデルと幾何設定 (Immutable)
# ==========================================
using LinearAlgebra, Rotations

# --- 形状パラメータ (太-細の非対称構成) ---
# z < 0 : Disk (太), z > 0 : Rod (細)
const FIXED_R_DISK = 0.020  # 円盤半径 20mm
const FIXED_L_DISK = 0.005  # 円盤厚 5mm (z: -5mm ~ 0mm)
const FIXED_R_ROD  = 0.0025 # ロッド半径 2.5mm
const FIXED_L_ROD  = 0.100  # ロッド長さ 100mm

# 設置角度
const FIXED_THETA = deg2rad(90.0)
const FIXED_PHI   = deg2rad(0.0)

# --- 遮蔽判定 (2段) ---
# 2つの円柱(Disk, Rod)のどちらかに隠れていれば遮蔽(true)
function is_blocked_2stage(pt, cam, r_disk, z_disk_min, z_disk_max, r_rod, z_rod_min, z_rod_max)
    if _check_cyl_block(pt, cam, r_disk, z_disk_min, z_disk_max) return true end
    if _check_cyl_block(pt, cam, r_rod, z_rod_min, z_rod_max) return true end
    return false
end

# 内部関数: 単一円柱の遮蔽チェック
function _check_cyl_block(pt, cam, r, z_min, z_max)
    cx, cy = cam[1], cam[2]; dx, dy = pt[1]-cx, pt[2]-cy
    a,b,c = dx^2+dy^2, 2*(cx*dx+cy*dy), cx^2+cy^2-r^2
    det = b^2-4*a*c
    if det < 0 return false end
    
    sqrt_det = sqrt(det)
    t1, t2 = (-b-sqrt_det)/(2a), (-b+sqrt_det)/(2a)
    t_ent, t_ext = min(t1,t2), max(t1,t2)
    
    dz = pt[3]-cam[3]
    if abs(dz)<1e-9
        if cam[3]<z_min || cam[3]>z_max return false end
        tz1, tz2 = -Inf, Inf
    else
        tz1, tz2 = (z_min-cam[3])/dz, (z_max-cam[3])/dz
    end
    
    ts, te = max(t_ent, min(tz1, tz2)), min(t_ext, max(tz1, tz2))
    return ts < te && max(ts, 0.01) < min(te, 0.99)
end