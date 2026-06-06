import os
import glob

base = os.path.dirname(os.path.abspath(__file__))
py_files = glob.glob(os.path.join(base, 'S3', '*.py'))
py_files += glob.glob(os.path.join(base, 'S4', '*.py'))

# Map of smart quotes to ASCII
replacements = {
    '“': '"',  # left double smart quote
    '”': '"',  # right double smart quote
    '‘': "'",  # left single smart quote
    '’': "'",  # right single smart quote
}

for fpath in py_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    changed = False
    for smart, ascii_q in replacements.items():
        if smart in content:
            content = content.replace(smart, ascii_q)
            changed = True

    if changed:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'  Fixed: {os.path.relpath(fpath, base)}')
    else:
        print(f'  Clean: {os.path.relpath(fpath, base)}')
