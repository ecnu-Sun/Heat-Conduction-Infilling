import os
import pathlib

# --- 配置 ---
# 要搜索的根目录，'.' 代表当前目录
ROOT_DIRECTORY = '.'
# 输出文件的名称
OUTPUT_FILENAME = 'code.txt'
# 要查找的文件扩展名
FILE_EXTENSION = '.py'
# --- 配置结束 ---

def export_project_code():
    """
    递归查找指定目录下的所有特定扩展名文件，
    并将它们的路径和内容写入到一个输出文件中。
    """
    print(f"开始在 '{os.path.abspath(ROOT_DIRECTORY)}' 目录中搜索 '{FILE_EXTENSION}' 文件...")
    
    # 获取当前脚本的绝对路径，以便在搜索结果中排除它自身
    try:
        script_self_path = pathlib.Path(__file__).resolve()
    except NameError:
        # 在交互式环境（如Jupyter）中运行时，__file__ 未定义
        script_self_path = pathlib.Path('export_code.py').resolve()

    # 使用 with 语句安全地打开输出文件，'w' 模式会覆盖旧文件
    # 使用 encoding='utf-8' 来支持中文等非英文字符
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
        # 使用 pathlib.Path.rglob 递归查找所有匹配的文件
        # .rglob('*' + FILE_EXTENSION) 会找到所有子目录中以 .py 结尾的文件
        found_files = sorted(list(pathlib.Path(ROOT_DIRECTORY).rglob('*' + FILE_EXTENSION)))
        
        if not found_files:
            message = f"未在 '{os.path.abspath(ROOT_DIRECTORY)}' 中找到任何 '{FILE_EXTENSION}' 文件。"
            print(message)
            outfile.write(message)
            return

        for file_path in found_files:
            # 检查是否是脚本自身，如果是则跳过
            if file_path.resolve() == script_self_path:
                print(f"    -> 跳过脚本自身: {file_path}")
                continue

            print(f"    -> 正在处理: {file_path}")
            
            # 写入一个清晰的分隔符和文件路径
            outfile.write("=" * 80 + "\n")
            outfile.write(f"文件路径 (Path): {file_path}\n")
            outfile.write("=" * 80 + "\n\n")
            
            try:
                # 读取文件内容并写入输出文件
                content = file_path.read_text(encoding='utf-8')
                outfile.write(content)
                outfile.write("\n\n") # 在文件内容后添加两个换行符，以分隔不同文件
            except Exception as e:
                error_message = f"[错误] 读取文件 {file_path} 时发生错误: {e}\n\n"
                print(error_message)
                outfile.write(error_message)

    print(f"\n处理完成！所有 '{FILE_EXTENSION}' 文件的内容已成功导出到 ./{OUTPUT_FILENAME}")

if __name__ == '__main__':
    export_project_code()