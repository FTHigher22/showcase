#reconst_voxcel.jl
using PythonCall
using JLD2
using LinearAlgebra
using Plots

gr()
cv = pyimport("cv2")
np = pyimport("numpy")

function main()
    @load "calibration_data.jld2" mtx1_jl dist1_jl mtx2_jl dist2_jl r1 t1 r2 t2
    @load "test.jld2" img1 img2

    # OpenCV用データ準備
    rvec1, tvec1 = np.array(r1), np.array(t1)
    rvec2, tvec2 = np.array(r2), np.array(t2)
    mtx1, dist1 = np.array(mtx1_jl), np.array(dist1_jl)
    mtx2, dist2 = np.array(mtx2_jl), np.array(dist2_jl)

    h, w = size(img1)
    
    # --- ボクセル空間設定 ---
    # 探索範囲 (放電サイズに合わせて調整してください)
    x_range = range(-2.0, 2.0, length=50)
    y_range = range(-2.0, 2.0, length=50)
    z_range = range(0.0, 8.0, length=100) # 高さ

    voxels = []
    
    # 全ボクセルを作成
    # 高速化のため、バッチ処理推奨ですが、分かりやすくループで記述
    # PythonCallのオーバーヘッド削減のため、ある程度まとめて投影します
    
    X_grid = [x for x in x_range, y in y_range, z in z_range]
    Y_grid = [y for x in x_range, y in y_range, z in z_range]
    Z_grid = [z for x in x_range, y in y_range, z in z_range]
    
    total_pts = length(X_grid)
    pts3d = zeros(total_pts, 3)
    pts3d[:, 1] = vec(X_grid)
    pts3d[:, 2] = vec(Y_grid)
    pts3d[:, 3] = vec(Z_grid)
    
    pts3d_np = np.array(pts3d, dtype=np.float64)

    println("ボクセル投影計算中... ($total_pts 点)")

    # 一括投影
    pts2d_1_np, _ = cv.projectPoints(pts3d_np, rvec1, tvec1, mtx1, dist1)
    pts2d_2_np, _ = cv.projectPoints(pts3d_np, rvec2, tvec2, mtx2, dist2)
    
    pts2d_1 = pyconvert(Array{Float32}, pts2d_1_np)
    pts2d_2 = pyconvert(Array{Float32}, pts2d_2_np)

    valid_voxels_x = Float64[]
    valid_voxels_y = Float64[]
    valid_voxels_z = Float64[]

    # 判定
    for i in 1:total_pts
        u1, v1 = Int(floor(pts2d_1[i, 1, 1])), Int(floor(pts2d_1[i, 1, 2]))
        u2, v2 = Int(floor(pts2d_2[i, 1, 1])), Int(floor(pts2d_2[i, 1, 2]))

        # 範囲チェック
        if 1 <= u1 < w && 1 <= v1 < h && 1 <= u2 < w && 1 <= v2 < h
            # 画素値チェック (閾値 0.5)
            # 両方のカメラで「白(放電)」である場合のみ残す
            if img1[v1, u1] > 0.5 && img2[v2, u2] > 0.5
                push!(valid_voxels_x, pts3d[i, 1])
                push!(valid_voxels_y, pts3d[i, 2])
                push!(valid_voxels_z, pts3d[i, 3])
            end
        end
    end

    println("有効ボクセル数: $(length(valid_voxels_x))")

    # プロット
    p1 = heatmap(img1, title="Cam 1", c=:grays, yflip=true, legend=false)
    p2 = heatmap(img2, title="Cam 2", c=:grays, yflip=true, legend=false)
    
    # 散布図としてボクセルを表示
    p3 = plot3d(valid_voxels_x, valid_voxels_y, valid_voxels_z, 
                seriestype=:scatter, markersize=2, markerstrokewidth=0,
                title="Voxel Carving", c=:cyan, alpha=0.6,
                xlabel="X", ylabel="Y", zlabel="Z")

    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
    savefig("result_voxel.png")
end

main()