# pdf2img.py —— 运行: python pdf2img.py "第二章 线性模型.pdf"
import fitz, sys, os

pdf_path = sys.argv[1]
out_dir = pdf_path.rsplit('.', 1)[0] + '_pages'
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)          # 150dpi 清晰度够读公式
    pix.save(f"{out_dir}/page_{i+1:03d}.png")
print(f"完成：{len(doc)} 页 → {out_dir}/")