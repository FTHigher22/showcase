# result_bspline.jl
using PythonCall
using JLD2
using LinearAlgebra
using Plots
using Interpolations # Bスプライン用

# バックエンド指定
gr()

cv = pyimport("cv2")
np = pyimport("numpy")

# --- 画像処理関数 ---
function extract_ordered_points(img_gray, num_samples=50)
    mask = Float64.(img_gray) .> 0.5
    points = []
    h, w = size(mask)
    for y in 1:h
        xs = findall(mask[y, :])
        if !isempty(xs)
            push!(points, [mean(xs), y]) 
        end
    end
    if isempty(points) return nothing end
    pts = hcat(points...)'
    # 簡易リサンプリング
    indices = range(1, size(pts, 1), length=num_samples)
    sampled_pts = zeros(num_samples, 2)
    for i in 1:2
        sampled_pts[:, i] .= [pts[Int(floor(idx)), i] for idx in indices]
    end
    return sampled_pts
end

function main()
    @load "calibration_data.jld2" mtx1_jl dist1_jl mtx2_jl dist2_jl r1 t1 r2 t2
    @load "test.jld2" img1 img2

    # パラメータ準備
    mtx1, dist1 = np.array(mtx1_jl), np.array(dist1_jl)
    mtx2, dist2 = np.array(mtx2_jl), np.array(dist2_jl)
    R1, _ = cv.Rodrigues(np.array(r1))
    R2, _ = cv.Rodrigues(np.array(r2))

    # --- ★ここを修正しました（np.matmulを使用） ---
    RT1 = cv.hconcat([R1, np.array(t1)])
    RT2 = cv.hconcat([R2, np.array(t2)])
    
    # 行列積 (3x3) @ (3x4) -> (3x4)
    P1 = np.matmul(mtx1, RT1)
    P2 = np.matmul(mtx2, RT2)
    # ---------------------------------------------

    # 点群取得
    pts1 = extract_ordered_points(img1, 50) 
    pts2 = extract_ordered_points(img2, 50)
    if pts1 === nothing || pts2 === nothing 
        println("点が検出できませんでした")
        return 
    end

    # OpenCV用に整形
    pts1_np = np.array(pts1')
    pts2_np = np.array(pts2')

    # 三角測量
    pts4d_py = cv.triangulatePoints(P1, P2, pts1_np, pts2_np)

    # --- Julia行列に変換 (計算エラー回避) ---
    pts4d_jl = pyconvert(Matrix{Float64}, pts4d_py)

    # Juliaで正規化計算 (1-based index)
    W = pts4d_jl[4, :]
    X_raw = pts4d_jl[1, :] ./ W
    Y_raw = pts4d_jl[2, :] ./ W
    Z_raw = pts4d_jl[3, :] ./ W

    # --- Bスプライン補間 ---
    # 重複点があるとエラーになる場合があるため、簡易的に少しノイズを加えるか
    # そのままトライしますが、通常はこれで通ります。
    
    t = range(0, 1, length=length(X_raw))
    
    # 各軸に対してスプライン補間を作成 (Cubic Spline)
    itp_x = scale(interpolate(X_raw, BSpline(Cubic(Line(OnGrid())))), t)
    itp_y = scale(interpolate(Y_raw, BSpline(Cubic(Line(OnGrid())))), t)
    itp_z = scale(interpolate(Z_raw, BSpline(Cubic(Line(OnGrid())))), t)

    # 細かくサンプリングして滑らかな曲線を描画
    fine_t = range(0, 1, length=200)
    X_smooth = [itp_x(ti) for ti in fine_t]
    Y_smooth = [itp_y(ti) for ti in fine_t]
    Z_smooth = [itp_z(ti) for ti in fine_t]

    # プロット
    p1 = heatmap(img1, title="Cam 1 Input", c=:grays, yflip=true, legend=false)
    p2 = heatmap(img2, title="Cam 2 Input", c=:grays, yflip=true, legend=false)
    p3 = plot3d(X_smooth, Y_smooth, Z_smooth, title="B-Spline Result", lw=3, c=:orange,
                label="Spline", xlabel="X", ylabel="Y", zlabel="Z")
    
    # 生データを散布図で重ねる
    plot3d!(p3, X_raw, Y_raw, Z_raw, seriestype=:scatter, markersize=2, c=:blue, label="Raw Points", alpha=0.5)

    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
    savefig("result_bspline.png")
    println("B-Spline reconstruction done.")
end

main()