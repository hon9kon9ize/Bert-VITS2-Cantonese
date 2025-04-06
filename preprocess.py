import argparse
from text.text_utils import TextCleaner
import ToJyutping
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="/notebooks/bert-vits2/dataset/zoengjyutgaai_saamgwokjinji/train.list",
    )
    parser.add_argument(
        "--output_file",
        default="/notebooks/bert-vits2/dataset/zoengjyutgaai_saamgwokjinji/train.list.phoneme",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep="|", header=None, names=["audio", "text"])
    text_cleaner = TextCleaner()

    for i, row in df.iterrows():
        text = row["text"]
        if not isinstance(text, str):
            print(f"Skipping row {i} due to invalid text format.")
            continue
        phoneme = ToJyutping.get_jyutping_text(text)
        phoneme_ids, _ = text_cleaner(phoneme)
        df.at[i, "phoneme"] = " ".join([str(i) for i in phoneme_ids])

    df[["audio", "phoneme"]].to_csv(
        args.output_file, sep="|", index=False, header=False
    )
