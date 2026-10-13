from pathlib import Path
p = Path('c:/Users/user/Desktop/IR作業/Hw-2/assignment_raw.html')
data = p.read_bytes()
for enc in ['utf-16','utf-8','utf-8-sig','cp950','big5','latin1']:
    try:
        s = data.decode(enc)
        if 'Word Bigram' in s or 'CKIP' in s or 'Wikipedia' in s:
            idx = s.find('Word Bigram') if 'Word Bigram' in s else s.find('CKIP') if 'CKIP' in s else s.find('Wikipedia')
            print('ENC', enc)
            print(s[idx-300:idx+5000])
            break
    except Exception:
        pass
