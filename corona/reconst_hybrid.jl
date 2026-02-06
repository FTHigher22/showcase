# result_hybrid.jl
using PythonCall
using JLD2
using LinearAlgebra
using Plots
using Statistics

gr()
cv = pyimport("cv2")
np = pyimport("numpy")

# --- 共通関数群 ---
function get_dist_map(img)
    binary = np.array(img .> 0.5, dtype=np.uint8)
    inv_binary = 1 .- binary
    dist_map = cv.distanceTransform(np.array(inv_binary, dtype=np.uint8), cv.DIST_L2, 5)
    return pyconvert(Matrix{Float32}, dist_map)
end

function main()
    @load "calibration_data.jld2" mtx1_jl dist1_jl mtx2_jl dist2_jl r1 t1 r2 t2
    @load "test.jld2" img1 img2

    rvec1, tvec1 = np.array(r1), np.array(t1)
    rvec2, tvec2 = np.array(r2), np.array(t2)
    mtx1, dist1 = np.array(mtx1_jl), np.array(dist1_jl)
    mtx2, dist2 = np.array(mtx2_jl), np.array(dist2_jl)
    
    h, w = size(img1)

    # --- Step 1: ボクセルカービング (粗い探索) ---
    println("Step 1: Voxel Carving (Initialization)...")
    # グリッドを少し粗めに設定して高速化
    x_range = range(-2.0, 2.0, length=30)
    y_range = range(-2.0, 2.0, length=30)
    z_range = range(0.0, 8.0, length=60)
    
    X_grid = [x for x in x_range, y in y_range, z in z_range]
    Y_grid = [y for x in x_range, y in y_range, z in z_range]
    Z_grid = [z for x in x_range, y in y_range, z in z_range]
    pts3d = hcat(vec(X_grid), vec(Y_grid), vec(Z_grid))
    pts3d_np = np.array(pts3d, dtype=np.float64)
    
    p1_np, _ = cv.projectPoints(pts3d_np, rvec1, tvec1, mtx1, dist1)
    p2_np, _ = cv.projectPoints(pts3d_np, rvec2, tvec2, mtx2, dist2)
    p1_proj = pyconvert(Array{Float32}, p1_np)
    p2_proj = pyconvert(Array{Float32}, p2_np)
    
    valid_indices = []
    for i in 1:size(pts3d, 1)
        u1, v1 = Int(floor(p1_proj[i,1,1])), Int(floor(p1_proj[i,1,2]))
        u2, v2 = Int(floor(p2_proj[i,1,1])), Int(floor(p2_proj[i,1,2]))
        if 1 <= u1 < w && 1 <= v1 < h && 1 <= u2 < w && 1 <= v2 < h
            if img1[v1, u1] > 0.5 && img2[v2, u2] > 0.5
                push!(valid_indices, i)
            end
        end
    end
    
    if isempty(valid_indices)
        println("ボクセルが見つかりませんでした。範囲を確認してください。")
        return
    end
    
    voxels = pts3d[valid_indices, :]

    # --- Step 2: ボクセルから初期スケルトン抽出 ---
    # Z座標ごとに重心を取ることでライン化する (放電がZ方向に伸びていると仮定)
    # 実際にはMST(最小全域木)等を使うのが一般的だが、ここではZスライスごとの平均を使う
    z_layers = sort(unique(voxels[:, 3]))
    skeleton_nodes = []
    for z in z_layers
        layer_mask = voxels[:, 3] .== z
        layer_pts = voxels[layer_mask, :]
        center = mean(layer_pts, dims=1)
        push!(skeleton_nodes, center)
    end
    nodes = vcat(skeleton_nodes...) # Nx3 Matrix

    # --- Step 3: Snakeによる微修正 ---
    println("Step 3: Snake Refinement...")
    dist1_map = get_dist_map(img1)
    dist2_map = get_dist_map(img2)
    
    num_nodes = size(nodes, 1)
    gamma = 20.0 # 引力
    
    for iter in 1:50 # イテレーション回数
        # 再投影
        pts_np = np.array(nodes, dtype=np.float64)
        pp1, _ = cv.projectPoints(pts_np, rvec1, tvec1, mtx1, dist1)
        pp2, _ = cv.projectPoints(pts_np, rvec2, tvec2, mtx2, dist2)
        pp1_arr = pyconvert(Array{Float32}, pp1)
        pp2_arr = pyconvert(Array{Float32}, pp2)
        
        forces = zeros(num_nodes, 3)
        delta = 0.02
        
        # 各ノードについて勾配降下
        for i in 1:num_nodes
            # 現在のコスト
            u1, v1 = pp1_arr[i,1,1], pp1_arr[i,1,2]
            u2, v2 = pp2_arr[i,1,1], pp2_arr[i,1,2]
            
            # コスト計算関数 (クロージャ)
            function calc_cost(p_test)
                p_np = np.array(p_test)
                pr1, _ = cv.projectPoints(p_np, rvec1, tvec1, mtx1, dist1)
                pr2, _ = cv.projectPoints(p_np, rvec2, tvec2, mtx2, dist2)
                pu1 = pyconvert(Vector{Float32}, pr1[0,0,:])
                pu2 = pyconvert(Vector{Float32}, pr2[0,0,:])
                c = 0.0
                if 1 <= pu1[1] < w && 1 <= pu1[2] < h
                    c += dist1_map[Int(floor(pu1[2])), Int(floor(pu1[1]))]
                else c += 50 end
                if 1 <= pu2[1] < w && 1 <= pu2[2] < h
                    c += dist2_map[Int(floor(pu2[2])), Int(floor(pu2[1]))]
                else c += 50 end
                return c
            end
            
            curr_c = calc_cost(nodes[i:i, :])
            
            # 数値微分
            grad = [0.0, 0.0, 0.0]
            for axis in 1:3
                nodes[i, axis] += delta
                new_c = calc_cost(nodes[i:i, :])
                grad[axis] = (new_c - curr_c) / delta
                nodes[i, axis] -= delta
            end
            
            forces[i, :] = -gamma * grad
        end
        
        # 更新とスムージング
        nodes += 0.1 * forces
        # 簡易スムージング
        for j in 2:num_nodes-1
            nodes[j, :] = 0.3*nodes[j-1, :] + 0.4*nodes[j, :] + 0.3*nodes[j+1, :]
        end
    end

    # --- プロット ---
    p1 = heatmap(img1, title="Cam 1", c=:grays, yflip=true, legend=false)
    p2 = heatmap(img2, title="Cam 2", c=:grays, yflip=true, legend=false)
    
    # ボクセルとSnake結果を重ねて表示
    p3 = plot3d(voxels[:,1], voxels[:,2], voxels[:,3], 
                seriestype=:scatter, markersize=1, alpha=0.1, c=:cyan, label="Voxel Hull")
    plot3d!(p3, nodes[:,1], nodes[:,2], nodes[:,3], lw=3, c=:magenta, label="Hybrid Snake",
            xlabel="X", ylabel="Y", zlabel="Z")

    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
    savefig("result_hybrid.png")
end

main()