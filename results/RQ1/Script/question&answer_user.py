import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# フォント設定（PDF用）
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 14


# ---------------------------------------------------
# 1. CSV読み込み
# ---------------------------------------------------
answers = pd.read_csv("user_answer_counts.csv")
questions = pd.read_csv("question_user_counts.csv")

# 列名統一
questions = questions.rename(columns={
    "owner_user_id": "user_id"
})


# ---------------------------------------------------
# 2. 回答・質問ユーザを結合
# ---------------------------------------------------
merged = pd.merge(
    answers,
    questions,
    on="user_id",
    how="outer"
)

# NaNを0に
merged = merged.fillna(0)


# ---------------------------------------------------
# 3. 回答上位20ユーザ
# ---------------------------------------------------
top_answer_users = merged.sort_values(
    by="total_answers",
    ascending=False
).head(20)


# ---------------------------------------------------
# 4. 質問上位20ユーザ
# ---------------------------------------------------
top_question_users = merged.sort_values(
    by="total_questions",
    ascending=False
).head(20)


# ---------------------------------------------------
# 5. 回答上位ユーザの積み上げグラフ
# ---------------------------------------------------
plt.figure(figsize=(12, 6))

plt.bar(
    top_answer_users["user_id"].astype(str),
    top_answer_users["total_answers"],
    label="Answers"
)

plt.bar(
    top_answer_users["user_id"].astype(str),
    top_answer_users["total_questions"],
    bottom=top_answer_users["total_answers"],
    label="Questions"
)

plt.xticks(rotation=90)
plt.xlabel("User ID")
plt.ylabel("Count")
plt.title("Top 20 Answer Users: Answers vs Questions")

plt.legend()
plt.tight_layout()

plt.savefig("top_answer_users_stacked.pdf")
plt.close()


# ---------------------------------------------------
# 6. 質問上位ユーザの積み上げグラフ
# ---------------------------------------------------
plt.figure(figsize=(12, 6))

plt.bar(
    top_question_users["user_id"].astype(str),
    top_question_users["total_questions"],
    label="Questions"
)

plt.bar(
    top_question_users["user_id"].astype(str),
    top_question_users["total_answers"],
    bottom=top_question_users["total_questions"],
    label="Answers"
)

plt.xticks(rotation=90)
plt.xlabel("User ID")
plt.ylabel("Count")
plt.title("Top 20 Question Users: Questions vs Answers")

plt.legend()
plt.tight_layout()

plt.savefig("top_question_users_stacked.pdf")
plt.close()


print("グラフ作成完了")