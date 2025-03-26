import os

def collect_md_filenames(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"指定的文件夹 {folder_path} 不存在。")
        return []

    md_filenames = []
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为 .md 文件
            if file.endswith('.md'):
                md_filenames.append(file)

    return md_filenames