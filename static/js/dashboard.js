document.addEventListener('DOMContentLoaded', function() {
    const dashboardToggleBtn = document.getElementById('dashboard-toggle-btn');
    const dashboardContent = document.getElementById('dashboard-content');

    let isDragging = false;
    let offsetX, offsetY;
    let hasMoved = false;

    // 点击切换显示
    dashboardToggleBtn.addEventListener('click', function(e) {
        if (hasMoved) {
            e.preventDefault();
            return;
        }
        if (dashboardContent.style.display === 'block') {
            setTimeout(() => {
                dashboardContent.style.display = 'none';
            }, 300); // 等待过渡动画结束后隐藏元素
            dashboardContent.classList.remove('active');
        } else {
            dashboardContent.style.display = 'block';
            // 使用setTimeout确保display:block先生效，再添加动画类
            setTimeout(() => {
                dashboardContent.classList.add('active');
            }, 10);
        }
    });

    // 拖动开始
    dashboardToggleBtn.addEventListener('mousedown', function(e) {
        isDragging = true;
        hasMoved = false;
        offsetX = e.clientX - dashboardToggleBtn.getBoundingClientRect().left;
        offsetY = e.clientY - dashboardToggleBtn.getBoundingClientRect().top;
        dashboardToggleBtn.style.position = 'absolute';
        dashboardToggleBtn.style.transition = 'none';
        dashboardToggleBtn.style.cursor = 'grabbing';
        e.preventDefault();
        e.stopPropagation();
    });

    // 拖动中
    document.addEventListener('mousemove', function(e) {
        if (isDragging) {
            hasMoved = true;
            let left = e.clientX - offsetX;
            let top = e.clientY - offsetY;

            // 限制不能超出屏幕边界
            const buttonSize = 40; // 按钮宽度
            const margin = 10; // 边缘安全距离

            left = Math.max(margin, Math.min(window.innerWidth - buttonSize - margin, left));
            top = Math.max(margin, Math.min(window.innerHeight - buttonSize - margin, top));

            dashboardToggleBtn.style.left = left + 'px';
            dashboardToggleBtn.style.top = top + 'px';
            localStorage.setItem('dashboardBtnPosition', JSON.stringify({ left, top }));
        }
    });

    // 拖动结束
    document.addEventListener('mouseup', function() {
        if (isDragging) {
            isDragging = false;
            dashboardToggleBtn.style.cursor = 'grab';
            dashboardToggleBtn.style.transition = '';
        }
    });

    // 初始化光标样式
    dashboardToggleBtn.style.cursor = 'grab';
    
    // 从 localStorage 加载保存的位置
    const savedPosition = JSON.parse(localStorage.getItem('dashboardBtnPosition'));
    if (savedPosition) {
        dashboardToggleBtn.style.position = 'absolute';
        dashboardToggleBtn.style.left = savedPosition.left + 'px';
        dashboardToggleBtn.style.top = savedPosition.top + 'px';
    }

    // 添加键盘快捷键 - 按Esc键重置按钮位置
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            resetButtonPosition();
        }
    });

    // 重置按钮位置的函数
    function resetButtonPosition() {
        dashboardToggleBtn.style.left = '';
        dashboardToggleBtn.style.top = '';
        // 应用默认位置
        dashboardToggleBtn.style.position = 'absolute';
        dashboardToggleBtn.style.transform = 'none';
    }

    // 页面加载时检查按钮是否在可视区域内，如果不在则重置
    function checkButtonVisibility() {
        const rect = dashboardToggleBtn.getBoundingClientRect();
        if (
            rect.left < 0 || 
            rect.right > window.innerWidth || 
            rect.top < 0 || 
            rect.bottom > window.innerHeight
        ) {
            resetButtonPosition();
        }
    }

    // 页面加载和调整大小时检查按钮位置
    checkButtonVisibility();
    window.addEventListener('resize', checkButtonVisibility);
});