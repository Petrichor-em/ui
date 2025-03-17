from flask import Flask, request, render_template, jsonify
# import lightrag_ollama
# from lightrag import QueryParam
import os
import markdown

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# rag = lightrag_ollama.rag

@app.route('/doc-lib')
def doc_lib():
    return render_template('doc-lib.html')

@app.route('/doc-view')
def doc_view():
    return render_template('doc-view.html')

@app.route('/ragquery', methods=['POST'])
def ragquery():
    data = request.get_json()
    input_message = data['data']
    return '{{reply}}'
#     response = rag.query(input_message, param=QueryParam(mode="global"))
#     return response

@app.route('/documents', methods=['POST'])
def documents():
    doc_path = request.get_json()['path']
    docs = get_files_info(doc_path)
    return docs

@app.route('/doc-content', methods=['POST'])
def doc_content():
    data = request.get_json()
    file_path = data.get('path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "文件不存在或路径无效"})
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 获取文件基本信息
        file_stats = os.stat(file_path)
        file_size = f'{(file_stats.st_size / 1024):.2f} KB'
        filename = os.path.basename(file_path)
        
        # 将Markdown转换为HTML
        html_content = markdown.markdown(
            content, 
            extensions=['fenced_code', 'tables']
        )
        
        return jsonify({
            "filename": filename,
            "path": file_path,
            "size": file_size,
            "content": html_content
        })
    
    except Exception as e:
        return jsonify({"error": f"读取文件失败: {str(e)}"})

def get_files_info(directory):
    files_info = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                files_info[file] = {
                    "path": os.path.join(root, file),
                    "size": f'{(os.path.getsize(os.path.join(root, file)) / 1024):.2f} KB'
            }
    return files_info


if __name__ == '__main__':
    app.run()