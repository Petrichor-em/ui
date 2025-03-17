document.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const docPath = urlParams.get('path');
    
    const documentContent = document.getElementById('documentContent');
    const docTitle = document.getElementById('docTitle');
    const docMeta = document.getElementById('docMeta');
    
    if (!docPath) {
        documentContent.innerHTML = '<div class="error">未找到文档路径</div>';
        return;
    }
    
    // 加载文档内容
    loadDocumentContent(docPath);
    
    async function loadDocumentContent(path) {
        try {
            const response = await fetch('/doc-content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ path: path })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                documentContent.innerHTML = `<div class="error">${data.error}</div>`;
                return;
            }
            
            // 设置文档标题
            docTitle.textContent = data.filename || '文档查看';
            
            // 设置文档元数据
            docMeta.innerHTML = `
                <div>路径: ${data.path}</div>
                <div>大小: ${data.size}</div>
            `;
            
            // 渲染文档内容
            documentContent.innerHTML = data.content;
            
            // 处理代码块的语法高亮（可选，如有需要可以添加库如highlight.js）
            
        } catch (error) {
            console.error("加载文档失败:", error);
            documentContent.innerHTML = '<div class="error">加载文档失败，请稍后再试</div>';
        }
    }
});
