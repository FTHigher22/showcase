# g_solve.jl
# ==========================================
# Part 3: 最適化実行 (Sharp Constraint for Shadow)
# ==========================================
include("cg_common.jl")
include("cg7_config_v4.jl")
include("g_plasma.jl") # ★v10を使用

using JLD2, Plots, Optim, LinearAlgebra, Rotations, ImageFiltering, Images, FileIO, Random

# 設定
const REAL_IMG1_PATH = "カメラの画像パス" 
const REAL_IMG2_PATH = "カメラの画像パス"

const FIXED_ROT_ANGLES = deg2rad.([-90.0, 0.0, 0.0])

const L_RES = 120
const L_RNG = 0.09

const local_grid = [[x, y, z] for z in range(-L_RNG,L_RNG,length=L_RES), 
                                  y in range(-L_RNG,L_RNG,length=L_RES), 
                                  x in range(-L_RNG,L_RNG,length=L_RES)] |> vec
const VOXEL_VOL = (2*L_RNG/(L_RES-1))^3
const VOXEL_STEP = 2*L_RNG/(L_RES-1)

# データ読み込み
println("Loading Images...")
const calib = load_calib_data()
r1 = Float64.(Gray.(load(REAL_IMG1_PATH)))
r2 = Float64.(Gray.(load(REAL_IMG2_PATH)))
img1_val = max.(r1 .- 0.1, 0.0)
img2_val = max.(r2 .- 0.1, 0.0)
norm_fac = max(maximum(img1_val), maximum(img2_val))
const img1_true = img1_val ./ norm_fac
const img2_true = img2_val ./ norm_fac
const H, W = size(img1_true)

# 重心計算
function get_centroid(img)
    total = sum(img)
    if total < 1e-9 return W/2, H/2 end
    sx, sy = 0.0, 0.0
    for j in 1:H, i in 1:W
        v = img[j,i]
        sx += i*v; sy += j*v
    end
    return sx/total, sy/total
end

function estimate_3d_position()
    u1, v1 = get_centroid(img1_true)
    u2, v2 = get_centroid(img2_true)
    function pos_loss(p)
        pt = [p[1], p[2], p[3]]
        u1e, v1e = project_point_fast(pt, calib.r1, calib.t1, calib.mtx1, calib.dist1)
        u2e, v2e = project_point_fast(pt, calib.r2, calib.t2, calib.mtx2, calib.dist2)
        if isnan(u1e) return 1e9 end
        return (u1-u1e)^2 + (v1-v1e)^2 + (u2-u2e)^2 + (v2-v2e)^2
    end
    res = optimize(pos_loss, [calib.board_w/2, calib.board_h/2, 0.3], NelderMead())
    return Optim.minimizer(res)
end

function generate_image_sharp(params_all)
    # [cx, cy, cz, d_rx, d_ry, d_rz, W_Rod, W_Disk, L_Rod, L_Disk, L_Radial, Gap_Rod, Gap_Disk]
    cx, cy, cz = params_all[1:3]
    d_rot = params_all[4:6]
    plasma_params = abs.(params_all[7:13]) 

    s1 = zeros(Float64, H, W)
    s2 = zeros(Float64, H, W)

    base_rot = RotXYZ(FIXED_ROT_ANGLES...)
    fine_rot = RotXYZ(d_rot...)
    R_obj = fine_rot * base_rot 
    
    center = [cx, cy, cz]
    cam1_loc = R_obj' * (calib.pos1 - center)
    cam2_loc = R_obj' * (calib.pos2 - center)
    
    disk_args = (FIXED_R_DISK*0.98, -FIXED_L_DISK, 0.0)
    rod_args  = (FIXED_R_ROD*0.90,  0.0, FIXED_L_ROD)

    Random.seed!(1234) 

    for pt_orig in local_grid
        jitter = (rand(3) .- 0.5) .* VOXEL_STEP
        pt_local = pt_orig .+ jitter
        
        r_curr = sqrt(pt_local[1]^2+pt_local[2]^2)
        # ★v10 (strict) を使用
        val = calc_plasma_density_strict(r_curr, pt_local[3], plasma_params)
        val *= VOXEL_VOL * 1e9
        if val < 1e-3 continue end

        blocked1 = is_blocked_2stage(pt_local, cam1_loc, disk_args..., rod_args...)
        if !blocked1
            pt_w = R_obj * pt_local + center
            u, v = project_point_fast(pt_w, calib.r1, calib.t1, calib.mtx1, calib.dist1)
            if 1<=u<W && 1<=v<H s1[round(Int,v), round(Int,u)] += val end
        end
        
        blocked2 = is_blocked_2stage(pt_local, cam2_loc, disk_args..., rod_args...)
        if !blocked2
            pt_w = R_obj * pt_local + center
            u, v = project_point_fast(pt_w, calib.r2, calib.t2, calib.mtx2, calib.dist1)
            if 1<=u<W && 1<=v<H s2[round(Int,v), round(Int,u)] += val end
        end
    end
    return s1, s2
end

function check_bounds(p)
    if any(abs.(p[4:6]) .> deg2rad(15.0)) return false end
    
    # 厚みを厳しく制限 (Max 5mm)
    if p[9] < 0.0001 || p[9] > 0.005 return false end # L_Rod
    if p[10]< 0.0001 || p[10]> 0.005 return false end # L_Disk
    
    if p[11] < 0.001 || p[11] > 0.050 return false end # Radial
    
    # Gap: Max 5mm
    if p[12] < 0.0 || p[12] > 0.005 return false end   
    if p[13] < 0.0 || p[13] > 0.005 return false end   
    return true
end

function loss_sse(p)
    if !check_bounds(p) return 1e9 end
    s1, s2 = generate_image_sharp(p)
    kern = Kernel.gaussian(2)
    s1_b, s2_b = imfilter(s1, kern), imfilter(s2, kern)
    t1_b, t2_b = imfilter(img1_true, kern), imfilter(img2_true, kern)
    max_s = max(maximum(s1_b), maximum(s2_b))
    if max_s > 1e-9 s1_b ./= max_s; s2_b ./= max_s end
    return sum((s1_b .- t1_b).^2) + sum((s2_b .- t2_b).^2)
end

function loss_cosine(p)
    if !check_bounds(p) return 1e9 end
    s1, s2 = generate_image_sharp(p)
    kern = Kernel.gaussian(2)
    s1_b, s2_b = imfilter(s1, kern), imfilter(s2, kern)
    n_s = norm(s1_b) + norm(s2_b)
    if n_s < 1e-9 return 1e9 end
    t1_b, t2_b = imfilter(img1_true, kern), imfilter(img2_true, kern)
    sim = dot(vec(s1_b), vec(t1_b))/(norm(s1_b)*norm(t1_b)) + 
          dot(vec(s2_b), vec(t2_b))/(norm(s2_b)*norm(t2_b))
    return -sim
end

# === 実行 ===

println("1. Estimating Position...")
est_pos = estimate_3d_position()
# [cx, cy, cz, d_rx, d_ry, d_rz, W_Rod, W_Disk, L_Rod, L_Disk, L_Radial, Gap_Rod, Gap_Disk]
p_init = [est_pos..., 0.0, 0.0, 0.0, 0.3, 2.0, 0.001, 0.001, 0.020, 0.001, 0.001] 

println("2. Stage 1: Coarse Optimization (SSE)...")
res1 = optimize(loss_sse, p_init, NelderMead(), Optim.Options(iterations=100, show_trace=true))
est1 = Optim.minimizer(res1)

println("3. Stage 2: Fine Tuning (Cosine)...")
res2 = optimize(loss_cosine, est1, NelderMead(), Optim.Options(iterations=250, show_trace=true))
est = Optim.minimizer(res2)

println("------------------------------------------------")
println("Final Estimate:")
println("  Pos : ", est[1:3])
println("  Fine Rot (deg): ", rad2deg.(est[4:6]))
println("  Weights: Rod=$(est[7]), Disk=$(est[8])")
println("  Thick  : Rod=$(est[9]*1000)mm, Disk=$(est[10]*1000)mm")
println("  Gap    : Rod=$(est[12]*1000)mm, Disk=$(est[13]*1000)mm")
println("------------------------------------------------")

s1, s2 = generate_image_sharp(est)
KERNEL_SHOW = Kernel.gaussian(2)
s1_show = imfilter(s1, KERNEL_SHOW)
max_s = maximum(s1_show)
if max_s > 0 s1_show ./= max_s end

p1 = heatmap(img1_true, title="True", aspect_ratio=:equal, yflip=true)
p2 = heatmap(s1_show, title="Est (Sharp L<3mm)", aspect_ratio=:equal, yflip=true)
p3 = heatmap(abs.(s1_show .- img1_true), title="Diff", aspect_ratio=:equal, yflip=true)
display(plot(p1, p2, p3, layout=(1,3), size=(1200, 400)))