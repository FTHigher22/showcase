# ==========================================
# Part 2: プラズマ発光モデル (Strict Hollow)
# ==========================================

# params: [W_Rod, W_Disk, L_Rod, L_Disk, L_Radial, Gap_Rod, Gap_Disk]
function calc_plasma_density_strict(r, z, params)
    W_Rod, W_Disk, L_Rod, L_Disk, L_Radial, Gap_Rod, Gap_Disk = params
    
    L_Rod  = max(L_Rod, 0.0001)
    L_Disk = max(L_Disk, 0.0001)
    L_Radial = max(L_Radial, 0.001)
    
    # 安全装置: 固体内部なら強制的にゼロ (影を確保)
    # ロッド内部
    if z > 0.0 && z <= FIXED_L_ROD && r < FIXED_R_ROD
        return 0.0 
    end
    # ディスク内部
    if z >= -FIXED_L_DISK && z <= 0.0 && r < FIXED_R_DISK
        return 0.0
    end
    
    dens_rod = 0.0
    dens_disk = 0.0
    
    # --- 1. ロッド由来 (Rod Glow) ---
    if z > 0.0 && z <= FIXED_L_ROD
        dist_from_rod = max(0.0, r - FIXED_R_ROD)
        dens_rod = exp(-(dist_from_rod - Gap_Rod)^2 / (2 * L_Rod^2))
    end
    
    # --- 2. ディスク由来 (Disk Glow) ---
    # A. 側面 (Side)
    if z >= -FIXED_L_DISK && z <= 0.0
        dist_side = max(0.0, r - FIXED_R_DISK)
        # ここは"Rod" のパラメータで光らせる
        dens_rod += exp(-(dist_side - Gap_Rod)^2 / (2 * L_Rod^2))
    end
    
    # B. 端面 (Faces)
    if r > FIXED_R_ROD && r < (FIXED_R_DISK + 0.005) 
        dist_face = min(abs(z - 0.0), abs(z - (-FIXED_L_DISK)))
        prof_z = exp(-(dist_face - Gap_Disk)^2 / (2 * L_Disk^2))
        
        dist_r = max(0.0, r - FIXED_R_ROD)
        prof_r = exp(-dist_r / L_Radial)
        
        dens_disk += prof_z * prof_r
    end
    
    return W_Rod * dens_rod + W_Disk * dens_disk
end