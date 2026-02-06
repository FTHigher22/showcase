using Images
using FileIO
using JLD2

# --- 1つ目の画像のパス入力 ---
print("1つ目の画像 (img1) のパスを入力してください: ")
path1 = strip(readline(), ['"', '\''])

# --- 2つ目の画像のパス入力 ---
print("2つ目の画像 (img2) のパスを入力してください: ")
path2 = strip(readline(), ['"', '\''])

# ファイルの存在確認
if isfile(path1) && isfile(path2)
    # 画像の読み込み（処理なしでそのまま格納）
    img1 = load(path1)
    img2 = load(path2)

    # 保存先のパス（スクリプトと同じディレクトリ）
    save_path = joinpath(@__DIR__, "test.jld2")

    # JLD2で保存（上書き）
    # 変数名 img1, img2 という名前で保存されます
    jldsave(save_path; img1, img2)

    println("✅ 2つの画像を $save_path に保存しました。")
    println("   - img1: $(size(img1))")
    println("   - img2: $(size(img2))")
else
    if !isfile(path1)
        println("⚠️ エラー: 1つ目のファイルが見つかりません: $path1")
    end
    if !isfile(path2)
        println("⚠️ エラー: 2つ目のファイルが見つかりません: $path2")
    end
end