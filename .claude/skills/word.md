---
name: word
description: Read, search, and edit Word (.docx) files. Use when the user asks to view, update, or modify .docx documents.
---

# Word 文档操控

使用 `python-docx` 库读写 `.docx` 文件。

## 安装

```bash
pip install python-docx -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 核心约定

- 所有脚本用 `python -c "..."` 单次执行
- 必须 `sys.stdout.reconfigure(encoding='utf-8')` 防止中文乱码
- 修改后必须 `doc.save()` 才生效
- Word 打开时文件被锁定，操作前先关掉 Word

## 操作脚本

### 查看结构 —— 列出所有标题及其行数

```
python -c "
from docx import Document
import sys
sys.stdout.reconfigure(encoding='utf-8')
doc = Document(r'<路径>')
cur = None; cnt = 0
for i, p in enumerate(doc.paragraphs):
    if p.style.name.startswith('Heading'):
        if cur: print(f'{cur}: {cnt} 行')
        cur = f'[{i}] ({p.style.name}) {p.text}'; cnt = 0
    elif p.text.strip(): cnt += 1
if cur: print(f'{cur}: {cnt} 行')
"
```

### 查看某段内容 —— 按行号区间输出

```
python -c "
from docx import Document
import sys
sys.stdout.reconfigure(encoding='utf-8')
doc = Document(r'<路径>')
for i in range(<起始>, min(<结束>+1, len(doc.paragraphs))):
    print(f'{i}: {doc.paragraphs[i].text}')
"
```

### 查找包含关键词的段落

```
python -c "
from docx import Document
import sys
sys.stdout.reconfigure(encoding='utf-8')
doc = Document(r'<路径>')
kw = '<关键词>'
for i, p in enumerate(doc.paragraphs):
    if kw in p.text: print(f'[{i}]: {p.text[:200]}')
"
```

### 用源文件更新标题下全部代码

找到 Heading 对应章节，删除旧段落，从 `.py` 源文件读入新代码逐行写入。

```
python -c "
from docx import Document
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

doc = Document(r'<docx路径>')
heading = '<标题文本>'
src = r'<源文件路径>'

# 定位标题
hi = None
for i, p in enumerate(doc.paragraphs):
    if p.style.name.startswith('Heading') and p.text.strip() == heading:
        hi = i; break
if hi is None: print(f'未找到: {heading}'); sys.exit(1)

# 下一标题 = 结束边界
nh = len(doc.paragraphs)
for i in range(hi+1, len(doc.paragraphs)):
    if doc.paragraphs[i].style.name.startswith('Heading'):
        nh = i; break

# 从后往前删旧段落
for i in range(nh-1, hi, -1):
    doc.paragraphs[i]._element.getparent().remove(doc.paragraphs[i]._element)

# 读源文件，逐行插入标题后
with open(src, 'r', encoding='utf-8') as f:
    lines = [l.rstrip('\n\r') for l in f.readlines()]

anchor = doc.paragraphs[hi]
for line in lines:
    np = doc.add_paragraph(line)
    anchor._element.addnext(np._element)
    anchor = np

doc.save(r'<docx路径>')
print(f'已更新 {heading}: {len(lines)} 行')
"
```

### 精确替换段落文本

```
python -c "
from docx import Document
import sys
sys.stdout.reconfigure(encoding='utf-8')
doc = Document(r'<路径>')
old = '<旧文本>'; new = '<新文本>'; n = 0
for p in doc.paragraphs:
    if p.text.strip() == old:
        for r in p.runs: r.text = ''
        if p.runs: p.runs[0].text = new
        else: p.text = new
        n += 1
doc.save(r'<路径>')
print(f'替换 {n} 处')
"
```

### 查看表格

```
python -c "
from docx import Document
import sys
sys.stdout.reconfigure(encoding='utf-8')
doc = Document(r'<路径>')
for ti, t in enumerate(doc.tables):
    print(f'=== 表{ti}: {len(t.rows)}行 x {len(t.columns)}列 ===')
    for ri, row in enumerate(t.rows):
        print(f'  R{ri}: {[c.text[:50] for c in row.cells]}')
"
```
