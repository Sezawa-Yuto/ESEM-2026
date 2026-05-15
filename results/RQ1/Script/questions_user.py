import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import matplotlib as mpl

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
# 6. 質問単位に変換（重複削除）
# ---------------------------------------------------
questions_df = df_final.drop_duplicates(subset="question_id").copy()

print("質問総数:", len(questions_df))


# ---------------------------------------------------
# 7. 質問ユーザごとの質問数
# ---------------------------------------------------
question_user_counts = (
    questions_df.groupby("owner_user_id")["question_id"]
    .count()
    .reset_index(name="total_questions")
)

total_users = question_user_counts["owner_user_id"].nunique()
print("質問ユーザ数:", total_users)


# ---------------------------------------------------
# 8. 質問数順ソート
# ---------------------------------------------------
question_user_counts = question_user_counts.sort_values(
    by="total_questions",
    ascending=False
).reset_index(drop=True)

print("最多質問ユーザ:")
print(question_user_counts.head())


# ---------------------------------------------------
# 9. 上位ユーザ可視化
# ---------------------------------------------------
TOP_N = 20
top_users = question_user_counts.head(TOP_N)

plt.figure(figsize=(12, 6))
plt.bar(
    top_users["owner_user_id"].astype(str),
    top_users["total_questions"]
)

plt.xticks(rotation=90)
plt.xlabel("User ID")
plt.ylabel("Number of Questions")
plt.title(f"Top {TOP_N} Question Askers")

plt.tight_layout()
plt.savefig("top_question_users.pdf")
plt.close()


# ---------------------------------------------------
# 10. ロングテール構造
# ---------------------------------------------------
question_user_counts["cumulative_ratio"] = (
    question_user_counts["total_questions"].cumsum()
    / question_user_counts["total_questions"].sum()
)

plt.figure(figsize=(8, 6))
plt.plot(
    np.arange(len(question_user_counts)),
    question_user_counts["cumulative_ratio"]
)

plt.xlabel("User Rank")
plt.ylabel("Cumulative Question Ratio")
plt.title("Cumulative Contribution (Questions)")

plt.tight_layout()
plt.savefig("cumulative_question_distribution.pdf")
plt.close()


# ---------------------------------------------------
# 11. 上位10%
# ---------------------------------------------------
top_10_percent = int(len(question_user_counts) * 0.1)

if top_10_percent < 1:
    top_10_percent = 1

ratio_10 = question_user_counts.iloc[top_10_percent - 1]["cumulative_ratio"]

print("上位10%の質問ユーザが占める割合:", ratio_10)


# ---------------------------------------------------
# 12. CSV保存
# ---------------------------------------------------
question_user_counts[["owner_user_id", "total_questions"]].to_csv(
    "question_user_counts.csv",
    index=False
)

print("分析完了")