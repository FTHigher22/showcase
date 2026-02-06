# reconst_snake.jl
using PythonCall
using JLD2
using LinearAlgebra
using Plots
using Statistics

# バックエンド指定
gr()

cv = pyimport("cv2")
np = pyimport("numpy")

function get_dist_map(img)
    # img: 0.0-1.0
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

    dist1_map = get_dist_map(img1)
    dist2_map = get_dist_map(img2)
    h, w = size(dist1_map)

    # 初期化
    num_nodes = 50
    nodes = zeros(num_nodes, 3)
    nodes[:, 1] .= 0.0 
    nodes[:, 2] .= 0.0 
    nodes[:, 3] = range(0, 8, length=num_nodes)

    # パラメータ
    gamma = 50.0  # 画像勾配への引力
    iterations = 100
    step_size = 0.05

    println("Snake最適化開始...")

    for iter in 1:iterations
        pts3d_np = np.array(nodes, dtype=np.float64)
        
        # --- 全点の一括投影 ---
        imgpts1_np, _ = cv.projectPoints(pts3d_np, rvec1, tvec1, mtx1, dist1)
        imgpts2_np, _ = cv.projectPoints(pts3d_np, rvec2, tvec2, mtx2, dist2)
        
        # Julia配列に変換
        imgpts1 = pyconvert(Array{Float32}, imgpts1_np)
        imgpts2 = pyconvert(Array{Float32}, imgpts2_np)

        forces = zeros(num_nodes, 3)

        for i in 1:num_nodes
            u1, v1 = imgpts1[i, 1, 1], imgpts1[i, 1, 2]
            u2, v2 = imgpts2[i, 1, 1], imgpts2[i, 1, 2]
            
            delta = 0.05
            grad = [0.0, 0.0, 0.0]
            
            # --- 【重要修正】安全装置付きコスト計算関数 ---
            function get_cost(u_val, v_val, dmap)
                # 1. NaN や Inf チェック
                if !isfinite(u_val) || !isfinite(v_val)
                    return 100.0
                end
                
                # 2. 値が巨大すぎる場合（10万画素以上とか）は強制的に画面外扱い
                # これにより Int64 変換エラーを防ぐ
                if abs(u_val) > 100000 || abs(v_val) > 100000
                    return 100.0 
                end

                # 3. 正常な場合のみ変換
                ix = Int(floor(u_val)) + 1
                iy = Int(floor(v_val)) + 1
                
                if 1 <= ix <= w && 1 <= iy <= h
                    return dmap[iy, ix]
                else
                    return 100.0 # 画面外ペナルティ
                end
            end
            # ----------------------------------------------

            current_cost = get_cost(u1, v1, dist1_map) + get_cost(u2, v2, dist2_map)

            # --- 微分ループ ---
            for axis in 1:3
                nodes[i, axis] += delta
                
                # 1点だけ再投影
                p_tmp = np.array(nodes[i:i, :]) 
                p1_n, _ = cv.projectPoints(p_tmp, rvec1, tvec1, mtx1, dist1)
                p2_n, _ = cv.projectPoints(p_tmp, rvec2, tvec2, mtx2, dist2)
                
                p1_arr = pyconvert(Array{Float32}, p1_n)
                p2_arr = pyconvert(Array{Float32}, p2_n)
                
                nu1, nv1 = p1_arr[1, 1, 1], p1_arr[1, 1, 2]
                nu2, nv2 = p2_arr[1, 1, 1], p2_arr[1, 1, 2]
                
                cost_new = get_cost(nu1, nv1, dist1_map) + get_cost(nu2, nv2, dist2_map)
                
                grad[axis] = (cost_new - current_cost) / delta
                nodes[i, axis] -= delta 
            end
            
            forces[i, :] = -gamma * grad 
        end
        
        # 更新
        nodes += step_size * forces
        
        # スムージング
        for j in 2:num_nodes-1
            nodes[j, :] = 0.25*nodes[j-1, :] + 0.5*nodes[j, :] + 0.25*nodes[j+1, :]
        end
    end

    p1 = heatmap(img1, title="Cam 1", c=:grays, yflip=true, legend=false)
    p2 = heatmap(img2, title="Cam 2", c=:grays, yflip=true, legend=false)
    p3 = plot3d(nodes[:,1], nodes[:,2], nodes[:,3], title="Snake Method", lw=3, c=:green,
                xlabel="X", ylabel="Y", zlabel="Z")
    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
    savefig("result_snake.png")
    println("Snake reconstruction done.")
end

main()