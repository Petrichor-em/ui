// 画布操作相关函数
function showCanvas() {
    const canvasContainer = document.querySelector('.canvas-container');
    canvasContainer.classList.add('visible');
    canvasContainer.classList.remove('collapsed');
    canvasContainer.style.height = '100vh';
    canvasContainer.style.transform = 'translateY(0)';
    updateToggleButton('fold canvas');
}

function hideCanvas() {
    const canvasContainer = document.querySelector('.canvas-container');
    canvasContainer.classList.remove('visible');
    canvasContainer.style.height = '60px';
    canvasContainer.style.transform = 'translateY(calc(100vh - 60px))';
    updateToggleButton('unfold canvas');
}

function toggleCanvas() {
    const canvasContainer = document.querySelector('.canvas-container');
    if (canvasContainer.classList.contains('collapsed')) {
        canvasContainer.classList.remove('collapsed');
        canvasContainer.style.height = '100vh';
        canvasContainer.style.transform = 'translateY(0)';
        updateToggleButton('fold canvas');
    } else {
        canvasContainer.classList.add('collapsed');
        canvasContainer.style.height = '60px';
        canvasContainer.style.transform = 'translateY(calc(100vh - 60px))';
        updateToggleButton('unfold canvas');
    }
}

function updateToggleButton(text) {
    const toggleBtn = document.querySelector('.canvas-toggle-btn .toggle-text');
    if (toggleBtn) {
        toggleBtn.textContent = text;
    }
}

const neo4jVisManager = {
    config: null,

    initialize: function(config) {
        this.config = config;
    },

    getConfig: function() {
        return this.config;
    }
};

// 画布内容管理器
const canvasManager = {
    // 清空画布
    clear: function() {
        const canvas = document.getElementById('background-canvas');
        canvas.innerHTML = '';
    },

    // 显示内容
    show: function(content, type = 'text') {
        this.clear();
        const canvas = document.getElementById('background-canvas');
        const element = document.createElement('div');
        
        element.style.cssText = `
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 100vh;
            padding: 0;
            margin: 0;
            border-radius: 8px;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        `;

        switch(type) {
            case 'image':
                element.innerHTML = `<img src="${content}" style="max-width: 100%; height: auto;">`;
                break;
            case 'html':
                element.innerHTML = content;
                break;
            case 'text':
                element.textContent = content;
                break;
            case 'graph':
                // 创建容器
                const container = document.createElement('div');
                container.style.cssText = `
                    position: absolute;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100vh;
                    display: flex;
                    padding: 0;
                    margin: 0;
                    background: white;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                `;

                // 创建图谱容器
                const graphContainer = document.createElement('div');
                graphContainer.id = 'neo4j-graph';
                graphContainer.style.cssText = `
                    width: 66.67%;
                    height: 100%;
                    position: relative;
                    background: #f8f8f8;
                    border-right: 1px solid #eee;
                    min-height: 800px;
                `;

                // 创建信息面板
                const infoPanel = document.createElement('div');
                infoPanel.id = 'info-panel';
                infoPanel.style.cssText = `
                    width: 33.33%;
                    height: 100%;
                    padding: 20px;
                    overflow-y: auto;
                    background: white;
                `;
                infoPanel.innerHTML = '<h3 style="color: #666;">点击节点或边查看详细信息</h3>';

                // 组装容器
                container.appendChild(graphContainer);
                container.appendChild(infoPanel);
                element.appendChild(container);
                canvas.appendChild(element);
                
                const config = neo4jVisManager.getConfig();
                
                setTimeout(() => {
                    try {
                        if (typeof NeoVis === 'undefined' && typeof neovis !== 'undefined') {
                            window.NeoVis = neovis;
                        }
                        
                        if (typeof NeoVis === 'undefined') {
                            throw new Error('NeoVis library not loaded');
                        }

                        const viz = new NeoVis.default({
                            container_id: 'neo4j-graph',
                            server_url: config.url,
                            server_user: config.user,
                            server_password: config.password,
                            encrypted: "ENCRYPTION_OFF",
                            trust: "TRUST_ALL_CERTIFICATES",
                            initial_cypher: config.initialCypher,
                            labels: {
                                "Node": {
                                    caption: function(node) {
                                        return node.properties.name || 
                                               node.properties.title || 
                                               node.properties.id || 
                                               node.labels[0] + '_' + node.id;
                                    }
                                }
                            },
                            visConfig: {
                                nodes: {
                                    shape: 'dot',
                                    scaling: {
                                        min: 20,
                                        max: 30,
                                        label: {
                                            enabled: true,
                                            min: 14,
                                            max: 18
                                        }
                                    },
                                    margin: 10,
                                    fixed: true
                                },
                                edges: {
                                    arrows: {
                                        to: { enabled: true, scaleFactor: 1 }
                                    },
                                    smooth: {
                                        type: 'continuous'
                                    }
                                },
                                physics: {
                                    enabled: true,
                                    barnesHut: {
                                        gravitationalConstant: -2000,
                                        centralGravity: 0.3,
                                        springLength: 200
                                    },
                                    minVelocity: 0.75,
                                    stabilization: {
                                        enabled: true,
                                        iterations: 100,
                                        updateInterval: 50
                                    }
                                },
                                interaction: {
                                    dragNodes: false,
                                    dragView: true,
                                    zoomView: true,
                                    hover: true,
                                    navigationButtons: true,
                                    keyboard: true
                                }
                            }
                        });

                        viz.registerOnEvent("clickNode", (node) => {
                            const infoPanel = document.getElementById('info-panel');
                            if (infoPanel) {
                                const nodeData = node.node || {};
                                const label = nodeData.label || '未知类型';
                                const rawData = nodeData.raw || {};

                                let html = `
                                    <h3 style="margin: 0 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #eee;">
                                        节点类型: ${label}
                                    </h3>
                                    <table style="width: 100%; border-collapse: collapse;">
                                        <tr style="background: #f5f5f5;">
                                            <th style="padding: 8px; text-align: left;">属性</th>
                                            <th style="padding: 8px; text-align: left;">值</th>
                                        </tr>
                                        <tr>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">ID</td>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">${nodeData.id}</td>
                                        </tr>
                                    `;

                                    if (rawData.properties) {
                                        Object.entries(rawData.properties).forEach(([key, value]) => {
                                            html += `
                                                <tr>
                                                    <td style="padding: 8px; border-bottom: 1px solid #eee;">${key}</td>
                                                    <td style="padding: 8px; border-bottom: 1px solid #eee;">${value}</td>
                                                </tr>
                                            `;
                                        });
                                    }
                                    
                                    html += '</table>';
                                    infoPanel.innerHTML = html;
                            }
                        });
                        
                        viz.registerOnEvent("clickEdge", (edge) => {
                            const infoPanel = document.getElementById('info-panel');
                            if (infoPanel) {
                                const edgeData = edge.edge || {};
                                const rawData = edgeData.raw || {};
                                const type = edgeData.label || rawData.type || '未知类型';

                                let html = `
                                    <h3 style="margin: 0 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #eee;">
                                        关系类型: ${type}
                                    </h3>
                                    <table style="width: 100%; border-collapse: collapse;">
                                        <tr style="background: #f5f5f5;">
                                            <th style="padding: 8px; text-align: left;">属性</th>
                                            <th style="padding: 8px; text-align: left;">值</th>
                                        </tr>
                                `;

                                if (edgeData.from && edgeData.to) {
                                    html += `
                                        <tr>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">From</td>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">${edgeData.from}</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">To</td>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">${edgeData.to}</td>
                                        </tr>
                                    `;
                                }

                                if (rawData.properties) {
                                    Object.entries(rawData.properties).forEach(([key, value]) => {
                                        html += `
                                            <tr>
                                                <td style="padding: 8px; border-bottom: 1px solid #eee;">${key}</td>
                                                <td style="padding: 8px; border-bottom: 1px solid #eee;">${value}</td>
                                            </tr>
                                        `;
                                    });
                                }
                                
                                html += '</table>';
                                infoPanel.innerHTML = html;
                            }
                        });

                        viz.render();

                    } catch (error) {
                        console.error("创建 Neo4j 可视化时出错:", error);
                        const errorDiv = document.createElement('div');
                        errorDiv.style.color = 'red';
                        errorDiv.style.padding = '20px';
                        errorDiv.textContent = `连接错误: ${error.message}`;
                        graphContainer.appendChild(errorDiv);
                    }
                }, 100);
                break;
            default:
                element.textContent = content;
        }

        canvas.appendChild(element);
        showCanvas();
    }
};

// 初始化画布控制
function initCanvasControls() {
    const toggleBtn = document.querySelector('.canvas-toggle-btn');
    const canvasContainer = document.querySelector('.canvas-container');
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleCanvas);
    }
    
    // 设置初始状态
    canvasContainer.classList.add('collapsed');
    updateToggleButton('unfold canvas');

    // 添加显示图谱按钮事件监听
    const showGraphBtn = document.getElementById('show-graph-btn');
    if (showGraphBtn) {
        showGraphBtn.addEventListener('click', function() {
            // 显示图谱
            canvasManager.show(null, 'graph');
            
            // 如果画布是折叠的，展开画布
            if (canvasContainer.classList.contains('collapsed')) {
                toggleCanvas();
            }
        });
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initCanvasControls();

    // 初始化 Neo4j 配置
    neo4jVisManager.initialize({
        url: "bolt://localhost:7687",
        user: "neo4j",
        password: "12345678",
        initialCypher: "MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100"
    });

    // 页面加载时不自动显示图谱，改为通过按钮触发
    // canvasManager.show(null, 'graph');  // 这行被移除
});