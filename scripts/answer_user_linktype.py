import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import matplotlib as mpl
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["font.size"] = 14


# ---------------------------------------------------
# 1. 年ごとの回答 CSV 読み込み
# ---------------------------------------------------
all_dfs = []

for year in range(2008, 2022):
    path = f"PerItemLinkCSV_{year}_filter/filtered_answers.csv"
    if os.path.exists(path):
        print(f"[LOAD] {year}")
        df_year = pd.read_csv(path, low_memory=False)
        df_year["year"] = year
        all_dfs.append(df_year)

if not all_dfs:
    raise ValueError("filtered_answers.csv が読み込めませんでした")

df = pd.concat(all_dfs, ignore_index=True)
total_questions_before = df["question_id"].nunique()
print("フィルタ前の質問総数:", total_questions_before)
print("総回答数（フィルタ前）:", len(df))


# ---------------------------------------------------
# 2. external_only とそれ以外に分割
# ---------------------------------------------------
df_external = df[df["link_type"] == "external_only"].copy()
df_other    = df[df["link_type"] != "external_only"].copy()


# ---------------------------------------------------
# 3. 主要外部ドメイン読み込み
# ---------------------------------------------------
domain_csv = "filtered_domains_ge_5pct_max2.csv"

selected_domains = (
    pd.read_csv(domain_csv)["domain"]
    .astype(str)
    .str.lower()
    .tolist()
)


# ---------------------------------------------------
# 4. external_only のみ主要ドメインでフィルタ
# ---------------------------------------------------
def contains_selected_domain(urls):
    if pd.isna(urls):
        return False
    urls = str(urls).lower()
    return any(dom in urls for dom in selected_domains)

df_external_filtered = df_external[
    df_external["urls"].apply(contains_selected_domain)
].copy()


# ---------------------------------------------------
# 5. 再結合（全カテゴリ対象）
# ---------------------------------------------------
df_final = pd.concat([df_other, df_external_filtered], ignore_index=True)

print("フィルタ後の総回答数:", len(df_final))
# ---------------------------------------------------
# 6. ユーザごとの回答数
# ---------------------------------------------------
user_counts = (
    df_final.groupby("user_id")["answer_id"]
    .count()
    .reset_index(name="total_answers")
)

# 回答数順ソート
user_counts = user_counts.sort_values(
    "total_answers",
    ascending=False
).reset_index(drop=True)

print("回答上位ユーザ")
print(user_counts.head())


# ---------------------------------------------------
# 7. 上位ユーザ取得
# ---------------------------------------------------
TOP_N = 50

top_users = user_counts.head(TOP_N)["user_id"]


# ---------------------------------------------------
# 8. 上位ユーザの回答抽出
# ---------------------------------------------------
df_top = df_final[df_final["user_id"].isin(top_users)].copy()


# ---------------------------------------------------
# 9. link_type集計
# ---------------------------------------------------
link_counts = (
    df_top.groupby(["user_id", "link_type"])
    .size()
    .unstack(fill_value=0)
)


# ---------------------------------------------------
# 10. 割合計算
# ---------------------------------------------------
link_ratio = link_counts.div(
    link_counts.sum(axis=1),
    axis=0
)


# ---------------------------------------------------
# ★重要：回答数順に並び替え
# ---------------------------------------------------
link_ratio = link_ratio.loc[top_users]


# ---------------------------------------------------
# 11. KMeansを用いたユーザーのクラスタリング
# ---------------------------------------------------
# link_ratio には各ユーザーの各link_typeの割合(合計1.0)が入っている
# クラスタ数(k)は仮で 4 に設定 (考察に合わせて 3〜5 などで調整してください)
NUM_CLUSTERS = 4

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
# link_ratio のデータを元にクラスタを予測
cluster_labels = kmeans.fit_predict(link_ratio)

# link_ratioのデータフレームにクラスタ情報を追加
link_ratio['Cluster'] = cluster_labels

# 各クラスタの中心（平均的なリンク割合）を計算して確認
cluster_centers = link_ratio.groupby('Cluster').mean()
print("【各クラスタの平均リンク割合】")
print(cluster_centers)

# クラスタごとにユーザーを並び替える（グラフを見やすくするため）
link_ratio_sorted = link_ratio.sort_values(by=['Cluster', 'user_id'])

# ---------------------------------------------------
# 12. クラスタリング結果の保存
# ---------------------------------------------------
link_ratio_sorted.to_csv("top_users_linktype_clusters.csv")

# ---------------------------------------------------
# 13. クラスタごとの積み上げグラフ (元のグラフの改良)
# ---------------------------------------------------
# Cluster列はグラフ描画から除外する
plot_data = link_ratio_sorted.drop(columns=['Cluster'])

ax = plot_data.plot(
    kind="bar",
    stacked=True,
    figsize=(24,10),
    colormap='viridis' # 色分けを見やすく変更
)

plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=12)

plt.xlabel("User ID (Sorted by Cluster)", fontsize=14)
plt.ylabel("Ratio", fontsize=14)
plt.title("Top Users Link Type Distribution by Cluster", fontsize=16)

# クラスタの境界線を引く（視覚的に分かりやすくするため）
current_x = 0
for cluster_id in range(NUM_CLUSTERS):
    cluster_size = sum(link_ratio_sorted['Cluster'] == cluster_id)
    current_x += cluster_size
    if current_x < len(link_ratio_sorted): # 最後には線を引かない
        plt.axvline(x=current_x - 0.5, color='red', linestyle='--', linewidth=2)

plt.legend(title="Link Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("top_users_linktype_clusters.pdf", bbox_inches="tight", dpi=300)
plt.close()

# ---------------------------------------------------
# 14. PCAによるクラスタの2次元散布図 (おまけ・全体像把握用)
# ---------------------------------------------------
# 割合データは多次元なので、PCAで2次元に圧縮して分布を見る
pca = PCA(n_components=2)
pca_result = pca.fit_transform(link_ratio.drop(columns=['Cluster']))

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=pca_result[:, 0], y=pca_result[:, 1], 
    hue=link_ratio['Cluster'], palette='Set1', s=100
)
plt.title("User Clusters visualized by PCA")
plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("cluster_pca_scatter.pdf", bbox_inches="tight", dpi=300)
plt.close()

print("クラスタリング分析完了")
# ---------------------------------------------------
# 11. (修正版) CSV保存
# ---------------------------------------------------
# クラスタリングに影響を与えないよう、出力用にコピーを作成
output_df = link_ratio.copy()

# 既に作られている user_counts (user_id と total_answers を持つ) をインデックスで結合
output_df = output_df.merge(
    user_counts.set_index("user_id")["total_answers"],
    left_index=True,
    right_index=True,
    how="left"
)

# 回答数 (total_answers) が多い順にソート
output_df = output_df.sort_values("total_answers", ascending=False)

# CSVとして保存
output_df.to_csv(
    "top_users_linktype_ratio.csv"
)
print("top_users_linktype_ratio.csv を保存しました。")


# ---------------------------------------------------
# 12. 積み上げグラフ
# ---------------------------------------------------
# グラフ化する際は回答数（total_answers）を除外して描画する
# （元のリンクカテゴリの割合だけを積み上げるため）
plot_data_ratio = output_df.drop(columns=["total_answers"])

plot_data_ratio.plot(
    kind="bar",
    stacked=True,
    figsize=(24,10)
)

plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=12)

plt.xlabel("User ID", fontsize=14)
plt.ylabel("Ratio", fontsize=14)
plt.title("Top Users Link Type Distribution", fontsize=16)

plt.legend(title="Link Type")
plt.tight_layout()
plt.savefig(
    "top_users_linktype_ratio.pdf",
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("上位ユーザ link_type 分析完了")