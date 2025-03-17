document.addEventListener('DOMContentLoaded', function() {
    const documentList = document.getElementById('documentList');
    const searchInput = document.getElementById('searchInput');
    
    let allDocuments = {};
    
    async function fetchDocuments() {
        try {
            const response = await fetch('/documents', {
                method: "POST",
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ path: 'docs' })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            allDocuments = await response.json();
            renderDocuments(allDocuments);
        } catch (error) {
            console.error("获取文档失败:", error);
            documentList.innerHTML = '<div class="no-results">加载文档失败，请稍后再试</div>';
        }
    }
    
    // 获取文件扩展名
    function getFileExtension(filename) {
        return filename.split('.').pop().toUpperCase();
    }
    
    // 渲染文档列表
    function renderDocuments(docs) {
        if (Object.keys(docs).length === 0) {
            documentList.innerHTML = '<div class="no-results">没有找到文档</div>';
            return;
        }
        
        let html = '';
        
        for (const [fileName, fileInfo] of Object.entries(docs)) {
            html += `
                <div class="document-item">
                    <a href="/doc-view?path=${encodeURIComponent(fileInfo.path)}">
                        <div class="doc-title">${fileName}</div>
                        <div class="doc-meta">
                            <span>类型: ${getFileExtension(fileName)}</span>
                            <span>大小: ${fileInfo.size}</span>
                        </div>
                    </a>
                </div>
            `;
        }
        
        documentList.innerHTML = html;
    }
    
    // 搜索文档
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        
        if (searchTerm.trim() === '') {
            renderDocuments(allDocuments);
            return;
        }
        
        const filteredDocs = {};
        
        for (const [fileName, fileInfo] of Object.entries(allDocuments)) {
            if (fileName.toLowerCase().includes(searchTerm)) {
                filteredDocs[fileName] = fileInfo;
            }
        }
        
        renderDocuments(filteredDocs);
    });

    // 初始化加载文档
    fetchDocuments();
});
