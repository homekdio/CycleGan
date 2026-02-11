import json

# 配置
source_file = "gradioui.py"        # 输入的 Python 文件
target_file = "gradioui_auto.ipynb" # 输出的 Notebook 文件

def convert_py_to_ipynb(src, dst):
    # 1. 读取 Python 源代码
    with open(src, 'r', encoding='utf-8') as f:
        # readlines() 会保留每一行的换行符，这正是 ipynb 格式需要的
        file_lines = f.readlines()

    # 2. 构建 Notebook 的 JSON 结构
    # .ipynb 文件本质上就是一个单纯的 JSON 文件
    notebook_content = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # 3. 核心逻辑：创建一个代码单元 (Code Cell)
    # 简单的做法是把所有代码塞进一个 cell
    # 进阶做法可以检测 "# %%" 分隔符来拆分多个 cell
    
    current_cell_lines = []
    
    # 这里我们演示如何把代码装进一个 cell 里
    main_code_cell = {
        "cell_type": "code",
        "execution_count": None, # null 代表还没运行过
        "metadata": {},
        "outputs": [],
        "source": file_lines
    }
    
    notebook_content["cells"].append(main_code_cell)

    # 4. 写入文件
    with open(dst, 'w', encoding='utf-8') as f:
        # indent=1 让生成的 JSON 有缩进，方便人类阅读
        json.dump(notebook_content, f, indent=1, ensure_ascii=False)
    
    print(f"✅转换成功: {src} -> {dst}")

if __name__ == "__main__":
    convert_py_to_ipynb(source_file, target_file)
