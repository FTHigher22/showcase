using PythonCall
using CondaPkg
using LinearAlgebra
using Statistics
using Plots
using JLD2

# Pythonライブラリ
cv = pyimport("cv2")
aruco = pyimport("cv2.aruco")
np = pyimport("numpy")

# --- 1. カメラ位置解析関数 (回転行列も返すように変更) ---
function get_camera_pose_data(video_path, mtx, dist, board, dictionary)
    cap = cv.VideoCapture(video_path)
    if !pyconvert(Bool, cap.isOpened())
        println("エラー: 動画が見つかりません -> $video_path")
        return nothing, nothing
    end

    rvecs_list = []
    tvecs_list = []
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    
    frame_count = 0
    max_frames = 500 

    while frame_count < max_frames
        ret, frame = cap.read()
        if !pyconvert(Bool, ret) break end

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if pyconvert(Any, ids) !== nothing && length(ids) > 0
            matched = board.matchImagePoints(corners, ids, nothing, nothing)
            objPoints, imgPoints = matched[0], matched[1]

            if length(objPoints) >= 4
                success, rvec, tvec = cv.solvePnP(objPoints, imgPoints, mtx, dist)
                if pyconvert(Bool, success)
                    push!(rvecs_list, pyconvert(Vector{Float64}, rvec.flatten()))
                    push!(tvecs_list, pyconvert(Vector{Float64}, tvec.flatten()))
                    frame_count += 1
                end
            end
        end
    end
    cap.release()

    if isempty(rvecs_list)
        return nothing, nothing
    end

    # 平均値を計算
    rvec_avg = mean(rvecs_list)
    tvec_avg = mean(tvecs_list)

    # 回転ベクトル -> 回転行列
    R_py, _ = cv.Rodrigues(np.array(rvec_avg))
    R = pyconvert(Matrix{Float64}, R_py)
    
    # カメラ位置 (World座標) C = -R^T * t
    cam_pos = -transpose(R) * tvec_avg

    # ★重要: 再構成には「カメラ位置(cam_pos)」と「ボードから見た回転(R)」の両方が役立ちます
    # ただし、OpenCVの projectPoints に渡すのは rvec_avg (World->Camera) と tvec_avg (World->Camera) です。
    # なので、これらを保存するのが最も間違いがありません。
    
    return rvec_avg, tvec_avg, cam_pos
end

function main_calc_and_save()
    # --- A. ボード設定 ---
    NX, NY = 1, 5
    MARKER_LEN = 0.034
    MARKER_SEP = 0.0098
    dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.GridBoard((NX, NY), Float64(MARKER_LEN), Float64(MARKER_SEP), dict)

    # --- B. カメラパラメータ ---
    mtx1 = np.array([2520.6977707322544 0.0 174.29321537559667; 0.0 2526.0917424509285 119.93257262136873; 0.0 0.0 1.0], dtype=np.float32)
    dist1 = np.array([-0.24573837136666832, 0.17785400980576882, -0.005573849617111708, -0.007868476583930702, 77.12717862091286], dtype=np.float32)
    
    mtx2 = np.array([2518.019765058265 0.0 238.19366807624564; 0.0 2523.5685461907888 171.46079787241402; 0.0 0.0 1.0], dtype=np.float32)
    dist2 = np.array([-0.17202579168341064, 0.6098408693367618, -0.003930808233794149, -0.004594410608522582, 0.016709559836629497], dtype=np.float32)


    # --- C. 動画パス ---
    video1 = "C:/JuliaProjects/3Dreconst/video/test160FPSaruco3_C001H001S0001.mp4" 
    video2 = "C:/JuliaProjects/3Dreconst/video/test160FPSaruco3_C001H002S0001.mp4" 

    println("計算開始...")
    
    # 計算実行
    r1, t1, pos1 = get_camera_pose_data(video1, mtx1, dist1, board, dict)
    r2, t2, pos2 = get_camera_pose_data(video2, mtx2, dist2, board, dict)

    if pos1 !== nothing && pos2 !== nothing
        println("計算成功。データを保存します...")

        # ★ Pythonオブジェクト(mtx, dist, board) はそのまま保存できない場合があるため、
        #   必要な数値データ（配列）として保存するか、または再構成側で作り直す設計にします。
        #   ここでは「カメラパラメータ(Matrix)」と「推定された外部パラメータ(Vector)」を保存します。
        
        # NumPy配列をJulia配列に変換して保存（安全策）
        mtx1_jl = pyconvert(Matrix{Float64}, mtx1)
        dist1_jl = pyconvert(Vector{Float64}, dist1)
        mtx2_jl = pyconvert(Matrix{Float64}, mtx2)
        dist2_jl = pyconvert(Vector{Float64}, dist2)

        # 保存
        @save "calibration_data.jld2" NX NY MARKER_LEN MARKER_SEP mtx1_jl dist1_jl mtx2_jl dist2_jl r1 t1 r2 t2 pos1 pos2

        println("保存完了: calibration_data.jld2")
        
        # プロット
        plotly() # インタラクティブモード
        
        # ボードを描画 (原点は左上)
        # ボード全体のサイズ（描画用）
        bw = (NX * MARKER_LEN + (NX-1) * MARKER_SEP)
        bh = (NY * MARKER_LEN + (NY-1) * MARKER_SEP)
        # GridBoardの座標系: X右, Y下, Z奥(または手前)
        board_x = [0, bw, bw, 0, 0]
        board_y = [0, 0, bh, bh, 0]
        board_z = [0, 0, 0, 0, 0]
        p = plot(board_x, board_y, board_z, 
                 label="Aruco Board", lw=3, color=:black, fill=(0, 0.2, :gray),
                 xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
                 title="Camera Positions (World Frame)",
                 aspect_ratio=:equal)
        # カメラ1
        scatter!(p, [pos1[1]], [pos1[2]], [pos1[3]], 
                 label="Camera 1", color=:blue, markersize=6, marker=:camera)
        # 原点と結ぶ線
        plot!(p, [0, pos1[1]], [0, pos1[2]], [0, pos1[3]], label="", ls=:dash, color=:blue)

        # カメラ2
        scatter!(p, [pos2[1]], [pos2[2]], [pos2[3]], 
                 label="Camera 2", color=:green, markersize=6, marker=:camera)
        # 原点と結ぶ線
        plot!(p, [0, pos2[1]], [0, pos2[2]], [0, pos2[3]], label="", ls=:dash, color=:green)
        
        # 原点マーク
        scatter!(p, [0], [0], [0], label="Origin (0,0,0)", color=:red, markersize=2)

        display(p)
    else
        println("失敗")
    end
end

main_calc_and_save()