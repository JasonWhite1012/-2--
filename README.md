# Information Retrieval Homework 2

這份專案依照投影片的需求，實作一個 word bigram language model。

## 目標

- 讀取 2023 中文維基百科 JSONL 語料
- 用中研院 CKIP 斷詞
- 先數出 unigram / bigram
- 計算模型機率
- 實作 `predict`, `prob`, `generate`
- 用小語料驗證結果

## 安裝

```bash
pip install -r requirements.txt
```

## 直接執行

```bash
python main.py --input data/wiki_article_list_2023_tra.json --max-articles 1000 --top-k 5
```

如果是 Windows PowerShell：

```powershell
python .\main.py --input .\data\wiki_article_list_2023_tra.json --max-articles 1000 --top-k 5
```

## 程式設計重點

- 一行是一篇文章，先做 JSON 讀取
- 用標點切句，每句前後加 `<s>` 與 `</s>`
- 標點與數字保留原樣
- 不要一次把整份 1.5 GB JSON 讀進記憶體
- 先用小語料測試，再跑完整語料

## 主要模組

- `src/wiki_bigram_lm.py`：模型與斷詞邏輯
- `main.py`：命令列入口
- `tests/test_wiki_bigram_lm.py`：小語料驗證

## CKIP

這份程式使用中研院 CKIP 的 Python 套件 `ckip-transformers`。若沒有安裝模組，程式會自動退回到簡單的空白切詞以便測試，但實際運行維基語料時仍建議安裝 CKIP。 
