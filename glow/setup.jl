using Pkg
Pkg.activate(".") # 現在のディレクトリ

# Juliaパッケージ
Pkg.add(["Plots", "JLD2", "LinearAlgebra", "Statistics", "Rotations", "Optim","ImageFiltering","PlotlyBase","FileIO", "Random"]) # 警告が出ていたので追加推奨
# Pkg.add(["PythonCall","JLD2", "CondaPkg"])
# using CondaPkg
# CondaPkg.add("python", version="=3.12")
# CondaPkg.add("numpy", version="<2")
# CondaPkg.add_pip("opencv-python")
# CondaPkg.status()


# Python環境 (CondaPkg) の設定
#using CondaPkg
#CondaPkg.add("python", version="=3.12")
#CondaPkg.add("numpy", version="<2")
#CondaPkg.add_pip("opencv-python") # pip経由でOpenCVを入れる