# reconst_linear.jl
using PythonCall
using JLD2
using LinearAlgebra
using Plots
using Statistics


gr()

# Pythonライブラリ
cv = pyimport("cv2")
np = pyimport("numpy")

# --- 画像処理関数 ---
function extract_ordered_points(img_gray, num_samples=100)
    # 二値化 (0.5を閾値とする)
    mask = Float64.(img_gray) .> 0.5
    
    points = []
    h, w = size(mask)
    for y in 1:h
        xs = findall(mask[y, :])
        if !isempty(xs)
            # 重心とY座標
            push!(points, [mean(xs), y]) 
        end
    end
    
    if isempty(points) return nothing end
    
    # 配列化 (Nx2)
    # hcat(points...) は 2xN になるので、転置(')して Nx2 にする
    pts = hcat(points...)' 
    
    # 簡易リサンプリング (Y軸方向に等間隔に間引く)
    indices = range(1, size(pts, 1), length=num_samples)
    sampled_pts = zeros(num_samples, 2)
    for i in 1:2
        sampled_pts[:, i] .= [pts[Int(floor(idx)), i] for idx in indices]
    end
    return sampled_pts
end

function main()
    # 1. データ読み込み
    @load "calibration_data.jld2" mtx1_jl dist1_jl mtx2_jl dist2_jl r1 t1 r2 t2
    @load "test.jld2" img1 img2

    # Numpy変換
    mtx1 = np.array(mtx1_jl)
    mtx2 = np.array(mtx2_jl)

    # 回転行列の作成
    R1, _ = cv.Rodrigues(np.array(r1))
    R2, _ = cv.Rodrigues(np.array(r2))

    # 射影行列 P = K [R|t] の作成
    # ★行列積は np.matmul を使用
    RT1 = cv.hconcat([R1, np.array(t1)])
    RT2 = cv.hconcat([R2, np.array(t2)])
    P1 = np.matmul(mtx1, RT1)
    P2 = np.matmul(mtx2, RT2)

    # 2. 画像処理 (放電の芯を抽出)
    pts1 = extract_ordered_points(img1, 200)
    pts2 = extract_ordered_points(img2, 200)

    if pts1 === nothing || pts2 === nothing
        println("画像から放電を検出できませんでした")
        return
    end

    # OpenCV用に形状変換
    # triangulatePoints は (2, N) の形状を要求するため転置する
    pts1_np = np.array(pts1')
    pts2_np = np.array(pts2')

    # 3. 三角測量 (Triangulation)
    # OpenCVの結果は 4xN の同次座標 (x, y, z, w)
    pts4d_py = cv.triangulatePoints(P1, P2, pts1_np, pts2_np)
    
    # --- Julia Matrix に変換してエラー回避 ---
    pts4d_jl = pyconvert(Matrix{Float64}, pts4d_py) 

    # --- 4. 同次座標 -> 3次元座標 (Juliaで計算) ---
    W = pts4d_jl[4, :]
    X = pts4d_jl[1, :] ./ W
    Y = pts4d_jl[2, :] ./ W
    Z = pts4d_jl[3, :] ./ W

    # 5. プロット
    p1 = heatmap(img1, title="Cam 1", c=:grays, yflip=true, aspect_ratio=:equal, legend=false)
    plot!(p1, pts1[:,1], pts1[:,2], w=2, c=:red)

    p2 = heatmap(img2, title="Cam 2", c=:grays, yflip=true, aspect_ratio=:equal, legend=false)
    plot!(p2, pts2[:,1], pts2[:,2], w=2, c=:red)

    p3 = plot3d(X, Y, Z, title="Linear Reconstruction", lw=2, c=:blue, 
                xlabel="X", ylabel="Y", zlabel="Z", legend=false)
    
    # 保存
    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
    savefig("result_linear.png")
    println("Linear reconstruction done.")
end

main()