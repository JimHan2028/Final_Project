# 从 v1 生成 v2 数据集，别反复跑，生成一次就够
# 跑法: python ml/make_v2_data.py
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "bank-additional-full.csv"
DST = ROOT / "data" / "bank-additional-v2.csv"


def main():
    df = pd.read_csv(SRC, sep=";")
    n0 = len(df)

    # 1. 教育程度归类：basic.4y/9y/6y 太碎了，合成一个 basic
    df["education"] = df["education"].replace(
        {"basic.4y": "basic", "basic.9y": "basic", "basic.6y": "basic"}
    )

    # 2. campaign 打了 20 次以上还没成交的属于骚扰级外呼，砍掉这几行
    df = df[df["campaign"] <= 20]

    # 3. 完全重复的行去掉
    df = df.drop_duplicates()

    df.to_csv(DST, sep=";", index=False)
    print(f"v1: {n0} rows -> v2: {len(df)} rows, saved to {DST}")
    print(f"education 合并后取值: {sorted(df['education'].unique())}")


if __name__ == "__main__":
    main()
