import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import matplotlib as mpl
import scipy.stats as stats

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
df_other = df[df["link_type"] != "external_only"].copy()


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

total_users = user_counts["user_id"].nunique()
print("対象ユーザ数:", total_users)


# ---------------------------------------------------
# 7. 回答数順ソート
# ---------------------------------------------------
user_counts = user_counts.sort_values(
    by="total_answers",
    ascending=False
).reset_index(drop=True)

print("最多回答ユーザ:")
print(user_counts.head())


# ---------------------------------------------------
# 8. 上位ユーザ可視化
# ---------------------------------------------------
TOP_N = 50

top_users_df = user_counts.head(TOP_N)
top_users = top_users_df["user_id"].tolist()

plt.figure(figsize=(12, 6))

plt.bar(
    top_users_df["user_id"].astype(str),
    top_users_df["total_answers"]
)

plt.xticks(rotation=90)

plt.xlabel("User ID")
plt.ylabel("Number of Answers")
plt.title(f"Top {TOP_N} Users (All Categories, external_only filtered)")

plt.tight_layout()
plt.savefig("top_users_all_categories_filtered.pdf")
plt.close()


# ---------------------------------------------------
# 9. ロングテール構造（累積割合）
# ---------------------------------------------------
user_counts["cumulative_ratio"] = (
    user_counts["total_answers"].cumsum()
    / user_counts["total_answers"].sum()
)

plt.figure(figsize=(8, 6))

plt.plot(
    np.arange(len(user_counts)),
    user_counts["cumulative_ratio"]
)

plt.xlabel("User Rank")
plt.ylabel("Cumulative Answer Ratio")
plt.title("Cumulative Contribution (All Categories)")

plt.tight_layout()
plt.savefig("cumulative_distribution_all_categories_filtered.pdf")
plt.close()


top_10_percent = int(len(user_counts) * 0.1)

if top_10_percent < 1:
    top_10_percent = 1

ratio_10 = user_counts.iloc[top_10_percent - 1]["cumulative_ratio"]

print("上位10%のユーザが占める回答割合:", ratio_10)
print("分析完了")


# ---------------------------------------------------
# ユーザ回答数CSV保存
# ---------------------------------------------------
user_counts[["user_id", "total_answers"]].to_csv(
    "user_answer_counts.csv",
    index=False
)


# ---------------------------------------------------
# 10. 上位ユーザの回答抽出
# ---------------------------------------------------
df_top = df_final[df_final["user_id"].isin(top_users)].copy()


# ---------------------------------------------------
# 11. ベストアンサー抽出
# ---------------------------------------------------
accepted = df_top[
    df_top["answer_id"] == df_top["accepted_answer_id"]
].copy()


# ---------------------------------------------------
# 12. ユーザごとの回答数
# ---------------------------------------------------
total_answers = (
    df_top.groupby("user_id")["answer_id"]
    .count()
)


# ---------------------------------------------------
# 13. ユーザごとのベストアンサー数
# ---------------------------------------------------
accepted_counts = (
    accepted.groupby("user_id")["answer_id"]
    .count()
)


# ---------------------------------------------------
# 14. データ結合
# ---------------------------------------------------
best_answer_stats = pd.DataFrame({
    "total_answers": total_answers,
    "accepted_answers": accepted_counts
}).fillna(0)


# ---------------------------------------------------
# 15. ベストアンサー率
# ---------------------------------------------------
best_answer_stats["accepted_ratio"] = (
    best_answer_stats["accepted_answers"]
    / best_answer_stats["total_answers"]
)


# 回答数順に並び替え
best_answer_stats = best_answer_stats.reindex(top_users)

print(best_answer_stats)


# ---------------------------------------------------
# 16. CSV保存
# ---------------------------------------------------
best_answer_stats.to_csv(
    "top_users_best_answer_stats.csv"
)


# ---------------------------------------------------
# 17. グラフ（件数）
# ---------------------------------------------------
plt.figure(figsize=(12,6))

plt.bar(
    best_answer_stats.index.astype(str),
    best_answer_stats["total_answers"],
    label="Total Answers"
)

plt.bar(
    best_answer_stats.index.astype(str),
    best_answer_stats["accepted_answers"],
    label="Accepted Answers"
)

plt.xticks(rotation=90)

plt.xlabel("User ID")
plt.ylabel("Count")
plt.title("Top Users: Accepted Answers")

plt.legend()

plt.tight_layout()
plt.savefig("top_users_accepted_answers.pdf")
plt.close()


# ---------------------------------------------------
# 18. グラフ（割合）
# ---------------------------------------------------
plt.figure(figsize=(12,6))

plt.bar(
    best_answer_stats.index.astype(str),
    best_answer_stats["accepted_ratio"]
)

plt.xticks(rotation=90)

plt.xlabel("User ID")
plt.ylabel("Accepted Answer Ratio")
plt.title("Top Users: Accepted Answer Ratio")

plt.tight_layout()
plt.savefig("top_users_accepted_ratio.pdf")
plt.close()


print("ベストアンサー分析完了")

# ---------------------------------------------------
# 19. ベストアンサー率の有意差検定と相関分析
# ---------------------------------------------------
print("\n=== ベストアンサーに関する相関と有意差分析 ===")

# 19.1 分析用のフラグ作成 (全データ df_final を対象)
# accepted_answer_id と answer_id が一致すればベストアンサー(1)、それ以外(0)
df_final["is_accepted"] = (df_final["answer_id"] == df_final["accepted_answer_id"]).astype(int)

# no_linkであるかどうかのフラグ(1=no_link, 0=それ以外)
df_final["is_no_link"] = (df_final["link_type"] == "no_link").astype(int)

print("\n=== ベストアンサーとなった4カテゴリの件数 ===")
# is_accepted が 1 (ベストアンサー) のデータだけを抽出し、link_type ごとにカウント
accepted_counts = df_final[df_final["is_accepted"] == 1]["link_type"].value_counts()
print(accepted_counts.to_string())
print("=============================================\n")
# ---------------------------------------------------
# 19.2 no_link vs リンクあり の有意差 (カイ二乗検定)
# ---------------------------------------------------
# クロス集計表の作成
crosstab_nolink = pd.crosstab(df_final["is_no_link"], df_final["is_accepted"])

chi2_val, p_val, dof, expected = stats.chi2_contingency(crosstab_nolink)
print("[1] no_link vs リンクあり のベストアンサー有意差 (カイ二乗検定)")
print(f"Chi-square: {chi2_val:.4f}, p-value: {p_val:.5e}")

if p_val < 0.05:
    print(" -> 結論: 'no_link' と 'リンクあり' でベストアンサーの選ばれやすさに統計的に有意な差があります。")
else:
    print(" -> 結論: 有意差は確認できませんでした。")


# ---------------------------------------------------
# 19.3 相関係数の算出 (点双列相関係数)
# ---------------------------------------------------
correlation, p_value_r = stats.pointbiserialr(df_final["is_no_link"], df_final["is_accepted"])
print(f"\n[2] 相関係数 (is_no_link と is_accepted): {correlation:.4f}")

if correlation < 0:
    print(" -> マイナスの相関: 'no_link' である（リンクがない）ほど、ベストアンサーに選ばれにくい傾向があります。")
elif correlation > 0:
    print(" -> プラスの相関: 'no_link' である（リンクがない）ほど、ベストアンサーに選ばれやすい傾向があります。")


# ---------------------------------------------------
# 19.4 4カテゴリすべてでの有意差と割合の比較
# ---------------------------------------------------
crosstab_all = pd.crosstab(df_final["link_type"], df_final["is_accepted"])
chi2_all, p_val_all, _, _ = stats.chi2_contingency(crosstab_all)

print("\n[3] 4つのカテゴリ間のベストアンサー有意差 (カイ二乗検定)")
print(f"Chi-square: {chi2_all:.4f}, p-value: {p_val_all:.5e}")

print("\n[参考] カテゴリ別のベストアンサー獲得率 (%)")
# 各カテゴリのベストアンサー率を計算 (1の平均値 * 100)
acceptance_rates = df_final.groupby("link_type")["is_accepted"].mean() * 100
print(acceptance_rates.sort_values(ascending=False))

# ---------------------------------------------------
# 上位ユーザのみ抽出
# ---------------------------------------------------
df_top_transition = df_final[
    df_final["user_id"].isin(top_users)
].copy()


# ---------------------------------------------------
# link_type 統一
# ---------------------------------------------------
df_top_transition["link_type"] = (
    df_top_transition["link_type"]
    .astype(str)
    .str.lower()
)


# ---------------------------------------------------
# 時系列ソート
# ---------------------------------------------------
df_top_transition["answer_date"] = pd.to_datetime(
    df_top_transition["answer_date"],
    errors="coerce"
)

df_top_sorted = df_top_transition.sort_values(
    ["user_id", "answer_date"]
)


# ---------------------------------------------------
# 前のカテゴリ
# ---------------------------------------------------
df_top_sorted["prev_link_type"] = (
    df_top_sorted.groupby("user_id")["link_type"]
    .shift(1)
)


# ---------------------------------------------------
# 遷移抽出
# ---------------------------------------------------
transitions_top = df_top_sorted.dropna(
    subset=["prev_link_type"]
).copy()


# ---------------------------------------------------
# 遷移カウント
# ---------------------------------------------------
transition_counts_top = (
    transitions_top
    .groupby(["prev_link_type", "link_type"])
    .size()
    .reset_index(name="count")
)


# ---------------------------------------------------
# 行列化
# ---------------------------------------------------
transition_matrix_top = transition_counts_top.pivot(
    index="prev_link_type",
    columns="link_type",
    values="count"
).fillna(0)


# ---------------------------------------------------
# 確率化
# ---------------------------------------------------
transition_prob_top = transition_matrix_top.div(
    transition_matrix_top.sum(axis=1),
    axis=0
)


# ---------------------------------------------------
# CSV保存
# ---------------------------------------------------
transition_matrix_top.to_csv("top_users_transition_counts.csv")
transition_prob_top.to_csv("top_users_transition_prob.csv")


# ---------------------------------------------------
# ヒートマップ
# ---------------------------------------------------
import seaborn as sns

plt.figure(figsize=(8,6))

sns.heatmap(
    transition_prob_top,
    annot=True,
    fmt=".2f",
    cmap="Reds"
)

plt.xlabel("Next Link Type")
plt.ylabel("Previous Link Type")
plt.title("Top Users: Link Type Transition")

plt.tight_layout()
plt.savefig("top_users_transition_heatmap.pdf")
plt.close()


print("上位ユーザ遷移分析 完了")
# ---------------------------------------------------
# 上位ユーザのカテゴリ別件数
# ---------------------------------------------------
link_counts_top = (
    df_top_transition["link_type"]
    .value_counts()
)

print("=== 上位ユーザ：カテゴリ別件数 ===")
print(link_counts_top)


# ---------------------------------------------------
# 対象ユーザ指定
# ---------------------------------------------------
TARGET_USER = 6309


# ---------------------------------------------------
# ユーザ抽出
# ---------------------------------------------------
df_user = df_final[
    df_final["user_id"] == TARGET_USER
].copy()

print("対象ユーザの回答数:", len(df_user))


# ---------------------------------------------------
# link_type 統一
# ---------------------------------------------------
df_user["link_type"] = (
    df_user["link_type"]
    .astype(str)
    .str.lower()
)


# ---------------------------------------------------
# 時系列ソート
# ---------------------------------------------------
df_user["answer_date"] = pd.to_datetime(
    df_user["answer_date"],
    errors="coerce"
)

df_user = df_user.sort_values("answer_date")


# ---------------------------------------------------
# 前のカテゴリ
# ---------------------------------------------------
df_user["prev_link_type"] = df_user["link_type"].shift(1)


# ---------------------------------------------------
# 遷移抽出
# ---------------------------------------------------
transitions_user = df_user.dropna(
    subset=["prev_link_type"]
).copy()


# ---------------------------------------------------
# 遷移カウント
# ---------------------------------------------------
transition_counts_user = (
    transitions_user
    .groupby(["prev_link_type", "link_type"])
    .size()
    .reset_index(name="count")
)

print("\n=== 遷移カウント ===")
print(transition_counts_user)


# ---------------------------------------------------
# 行列化
# ---------------------------------------------------
transition_matrix_user = transition_counts_user.pivot(
    index="prev_link_type",
    columns="link_type",
    values="count"
).fillna(0)

print("\n=== 遷移行列 ===")
print(transition_matrix_user)


# ---------------------------------------------------
# 確率化
# ---------------------------------------------------
transition_prob_user = transition_matrix_user.div(
    transition_matrix_user.sum(axis=1),
    axis=0
)

print("\n=== 遷移確率 ===")
print(transition_prob_user)


# ---------------------------------------------------
# ヒートマップ
# ---------------------------------------------------
import seaborn as sns

plt.figure(figsize=(6,5))

sns.heatmap(
    transition_prob_user,
    annot=True,
    fmt=".2f",
    cmap="Blues"
)

plt.xlabel("Next Link Type")
plt.ylabel("Previous Link Type")
plt.title(f"User {TARGET_USER}: Link Type Transition")

plt.tight_layout()
plt.savefig(f"user_{TARGET_USER}_transition_heatmap.pdf")
plt.close()


print("個別ユーザ遷移分析 完了")

df_user["answer_date"] = pd.to_datetime(
    df_user["answer_date"],
    errors="coerce"
)

# NaT削除
df_user = df_user.dropna(subset=["answer_date"])

# タイムゾーン削除
df_user["answer_date"] = df_user["answer_date"].dt.tz_localize(None)

# ソート
df_user = df_user.sort_values("answer_date")

mapping = {
    "no_link": 0,
    "internal_only": 1,
    "external_only": 2,
    "both": 3
}

df_user["link_num"] = df_user["link_type"].map(mapping)

plt.figure(figsize=(12,3))

plt.plot(
    df_user["answer_date"],
    df_user["link_num"],
    marker="o"
)

plt.yticks(
    [0,1,2,3],
    ["No Link", "Internal", "External", "Both"]
)

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("user_timeline_fixed.pdf")
plt.close()