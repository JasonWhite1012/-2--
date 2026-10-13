from __future__ import annotations

import argparse
import time

from src.wiki_bigram_lm import CKIPTokenizer, WikiBigramLM, resolve_input_path


def build_model(input_path: str, max_articles: int | None = None) -> tuple[WikiBigramLM, float]:
    input_file = resolve_input_path(input_path)
    tokenizer = CKIPTokenizer()
    model = WikiBigramLM(tokenizer=tokenizer)

    start = time.perf_counter()

    def progress_callback(processed: int, pct: float) -> None:
        print(f"進度: {processed} / {max_articles if max_articles is not None else '全部'} ({pct:.1f}%)", end="\r")

    model.fit_file(input_file, limit=max_articles, progress_callback=progress_callback)
    elapsed = time.perf_counter() - start
    print()
    return model, elapsed


def interactive_menu(model: WikiBigramLM) -> None:
    while True:
        print("\n=== Word Bigram Menu ===")
        print("1. predict(w, k)")
        print("2. prob(sentence)")
        print("3. generate(seed, n)")
        print("4. exit")
        choice = input("請選擇功能 (1-4): ").strip()

        if choice == "1":
            word = input("請輸入要預測的詞: ").strip()
            top_k = input("請輸入 top-k (預設 5): ").strip() or "5"
            print(f"\npredict({word!r}, {top_k})")
            for token, prob in model.predict(word, top_k=int(top_k)):
                print(f"  {word} -> {token}: {prob:.6f}")

        elif choice == "2":
            sentence = input("請輸入句子: ").strip()
            score = model.prob(sentence)
            print(f"\nprob({sentence!r}) = {score}")

        elif choice == "3":
            seed = input("請輸入 seed word: ").strip()
            length = input("請輸入生成長度 (預設 10): ").strip() or "10"
            sample = input("要抽樣生成嗎? (y/N): ").strip().lower() in {"y", "yes"}
            result = model.generate(seed, length=int(length), sample=sample)
            print(f"\ngenerate({seed!r}, {length}, sample={sample})")
            print(" ".join(result))

        elif choice in {"4", "q", "quit", "exit"}:
            print("已離開。")
            break

        else:
            print("無效選項，請輸入 1, 2, 3 或 4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Word bigram language model with CKIP segmentation")
    parser.add_argument("--input", type=str, default="wiki_article_list_2023_tra.json", help="預設固定使用 wiki_article_list_2023_tra.json")
    parser.add_argument("--max-articles", type=int, default=None, help="若不指定，則讀完整份語料；若要測試，輸入例如 1000")
    parser.add_argument("--top-k", type=int, default=5, help="predict 會列出的前 k 個詞")
    parser.add_argument("--seed", type=str, default="大學", help="generate 的起始詞")
    parser.add_argument("--length", type=int, default=10, help="generate 的長度")
    parser.add_argument("--sample", action="store_true", help="generate 時抽樣而非貪婪選擇")
    parser.add_argument("--interactive", action="store_true", help="進入互動式選單")
    args = parser.parse_args()

    input_path = args.input
    print(f"正在載入語料: {resolve_input_path(input_path)}")
    if args.max_articles is not None:
        print(f"讀取文章數: {args.max_articles}")
    else:
        print("讀取文章數: 全部語料")

    model, elapsed = build_model(input_path, max_articles=args.max_articles)
    print(f"Using corpus: {resolve_input_path(input_path)}")
    print(f"N = {model.N}")
    print(f"V = {model.V}")
    print(f"B = {model.B}")
    print(f"t = {elapsed:.2f}s")

    if args.interactive:
        interactive_menu(model)
        return

    print("\nPredict:")
    for token, prob in model.predict(args.seed, top_k=args.top_k):
        print(f"  {args.seed} -> {token}: {prob:.6f}")

    print("\nGenerate:")
    print(" ".join(model.generate(args.seed, length=args.length, sample=args.sample)))


if __name__ == "__main__":
    main()
