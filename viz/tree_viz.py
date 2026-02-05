"""
트리 구조 시각화 모듈

트리 구조를 그래프 형태로 시각화:
- b (base): 빨간색 - 루트
- s (step): 하늘색 - 단차
- g (groove): 노란색 - 홈
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================================
# 트리 그래프 레이아웃
# ============================================================================

def hierarchy_layout(
    nodes: List[Dict],
    width: float = 1.0,
    vert_gap: float = 0.25,
    vert_loc: float = 0.0,
    xcenter: float = 0.5
) -> Dict[int, Tuple[float, float]]:
    """
    계층적 트리 레이아웃 계산.
    
    Args:
        nodes: 노드 딕셔너리 리스트
        width: 전체 폭
        vert_gap: 레벨 간 간격
        vert_loc: 루트 y 위치
        xcenter: 루트 x 위치
        
    Returns:
        {node_id: (x, y)} 좌표 딕셔너리
    """
    # 노드 ID → 노드 데이터 매핑
    node_map = {n["id"]: n for n in nodes}
    
    # 루트 찾기
    root_id = None
    for n in nodes:
        if n["parent"] is None:
            root_id = n["id"]
            break
    
    if root_id is None:
        return {}
    
    pos = {}
    
    def _layout(node_id: int, x: float, y: float, w: float):
        """재귀적으로 레이아웃 계산"""
        pos[node_id] = (x, y)
        
        node = node_map[node_id]
        children = node.get("children", [])
        
        if not children:
            return
        
        n_children = len(children)
        child_width = w / n_children
        
        for i, child_id in enumerate(children):
            child_x = x - w / 2 + child_width / 2 + i * child_width
            _layout(child_id, child_x, y - vert_gap, child_width)
    
    _layout(root_id, xcenter, vert_loc, width)
    return pos


# ============================================================================
# 단일 트리 시각화
# ============================================================================

def visualize_tree(
    tree: Dict,
    ax: plt.Axes = None,
    title: str = None,
    node_size: int = 800,
    font_size: int = 14,
    show_id: bool = False
):
    """
    단일 트리 시각화.
    
    Args:
        tree: 트리 딕셔너리
        ax: matplotlib Axes (없으면 새로 생성)
        title: 그래프 제목
        node_size: 노드 크기
        font_size: 라벨 폰트 크기
        show_id: 노드 ID 표시 여부
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    nodes = tree["nodes"]
    canonical = tree.get("canonical", "")
    
    # 최대 깊이 계산
    max_depth = max(n.get("depth", 0) for n in nodes) if nodes else 0
    vert_gap = 0.2
    
    # 레이아웃 계산 (깊이에 맞게 조정)
    pos = hierarchy_layout(nodes, vert_gap=vert_gap)
    
    # 색상 맵 (step: 하늘색, groove: 노란색)
    color_map = {
        'b': '#FF6B6B',  # 빨간색 - base
        's': '#87CEEB',  # 하늘색 - step
        'g': '#FFE66D'   # 노란색 - groove
    }
    
    # 엣지 그리기
    for node in nodes:
        if node["parent"] is not None:
            parent_pos = pos[node["parent"]]
            node_pos = pos[node["id"]]
            ax.plot(
                [parent_pos[0], node_pos[0]],
                [parent_pos[1], node_pos[1]],
                color='#666666', linewidth=2, zorder=1
            )
    
    # 노드 그리기
    for node in nodes:
        x, y = pos[node["id"]]
        color = color_map.get(node["label"], '#888888')
        
        # 노드 원
        circle = plt.Circle((x, y), 0.04, color=color, ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        # 라벨
        label = node["label"]
        if show_id:
            label = f"{label}\n({node['id']})"
        
        ax.text(x, y, label, ha='center', va='center',
               fontsize=font_size, fontweight='bold', zorder=3)
    
    # 설정 (y축 범위를 깊이에 맞게 동적 조정)
    y_min = -max_depth * vert_gap - 0.1  # 여유 공간 추가
    y_max = 0.15
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 제목
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(canonical, fontsize=10, fontweight='bold')


# ============================================================================
# 여러 트리 시각화 (그리드)
# ============================================================================

def visualize_trees_grid(
    trees: List[Dict],
    n_cols: int = 4,
    figsize: Tuple[int, int] = None,
    output_path: str = None,
    show_legend: bool = True,
    suptitle: str = None
) -> Optional[plt.Figure]:
    """
    여러 트리를 그리드 형태로 시각화.
    
    Args:
        trees: 트리 딕셔너리 리스트
        n_cols: 열 개수
        figsize: 그림 크기
        output_path: 저장 경로 (None이면 저장 안함)
        show_legend: 범례 표시 여부
        suptitle: 전체 제목
        
    Returns:
        matplotlib Figure (또는 저장 시 None)
    """
    n_trees = len(trees)
    if n_trees == 0:
        return None
    
    n_rows = (n_trees + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # axes를 1D 배열로 변환 (모든 경우를 처리)
    if n_rows == 1 and n_cols == 1:
        # 단일 subplot인 경우 axes가 단일 객체로 반환됨
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        # 1행 또는 1열인 경우 axes가 1D 배열로 반환됨
        axes = np.atleast_1d(axes)
    else:
        # 2D 배열인 경우 flatten
        axes = axes.flatten()
    
    for i, tree in enumerate(trees):
        visualize_tree(tree, ax=axes[i])
    
    # 남은 축 숨기기
    for i in range(n_trees, len(axes)):
        axes[i].axis('off')
    
    # 범례 (step: 하늘색, groove: 노란색)
    if show_legend:
        legend_elements = [
            mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='b (base/root)'),
            mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='s (step)'),
            mpatches.Patch(facecolor='#FFE66D', edgecolor='black', label='g (groove)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3,
                  fontsize=10, bbox_to_anchor=(0.5, 0.98))
    
    # 전체 제목
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.0)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  트리 시각화 저장: {output_path}")
        return None
    
    return fig


# ============================================================================
# 트리 통계 시각화
# ============================================================================

def visualize_tree_statistics(
    trees: List[Dict],
    output_path: str = None
) -> Optional[plt.Figure]:
    """
    트리 통계 시각화 (step/groove 분포).
    
    Args:
        trees: 트리 딕셔너리 리스트
        output_path: 저장 경로
        
    Returns:
        matplotlib Figure (또는 저장 시 None)
    """
    # 통계 계산
    stats = []
    for tree in trees:
        nodes = tree["nodes"]
        s_count = sum(1 for n in nodes if n["label"] == "s")
        g_count = sum(1 for n in nodes if n["label"] == "g")
        stats.append((s_count, g_count))
    
    s_counts = [s for s, g in stats]
    g_counts = [g for s, g in stats]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Step 분포 (하늘색)
    s_unique, s_freq = np.unique(s_counts, return_counts=True)
    axes[0].bar(s_unique, s_freq, color='#87CEEB', edgecolor='black')
    axes[0].set_xlabel('Step Count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Step Node Distribution')
    axes[0].set_xticks(s_unique)
    
    # Groove 분포 (노란색)
    g_unique, g_freq = np.unique(g_counts, return_counts=True)
    axes[1].bar(g_unique, g_freq, color='#FFE66D', edgecolor='black')
    axes[1].set_xlabel('Groove Count')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Groove Node Distribution')
    axes[1].set_xticks(g_unique)
    
    # Step vs Groove 산점도
    # 각 (s, g) 조합의 빈도 계산
    from collections import Counter
    combo_counts = Counter(stats)
    
    for (s, g), count in combo_counts.items():
        size = count * 100
        axes[2].scatter(s, g, s=size, c='#FF6B6B', edgecolors='black', alpha=0.7)
        axes[2].annotate(f'{count}', (s, g), ha='center', va='center', fontsize=8)
    
    axes[2].set_xlabel('Step Count')
    axes[2].set_ylabel('Groove Count')
    axes[2].set_title('Step vs Groove Distribution')
    axes[2].set_xticks(range(max(s_counts) + 1))
    axes[2].set_yticks(range(max(g_counts) + 1))
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f'Tree Structure Statistics (N={len(trees)} trees)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  트리 통계 시각화 저장: {output_path}")
        return None
    
    return fig


# ============================================================================
# 편의 함수
# ============================================================================

def visualize_trees(
    trees: List[Dict],
    output_dir: str = None,
    prefix: str = "trees",
    max_display: int = 4
) -> None:
    """
    트리 리스트 시각화 및 저장 (그리드만).
    
    Args:
        trees: 트리 딕셔너리 리스트
        output_dir: 출력 디렉토리
        prefix: 파일명 접두사
        max_display: 그리드에 표시할 최대 트리 수 (기본: 4)
    """
    if not trees:
        print("  시각화할 트리가 없습니다.")
        return
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 그리드 시각화만 저장
        display_trees = trees[:max_display]
        n_nodes = trees[0].get("N", "?")
        max_depth = trees[0].get("max_depth_constraint", "?")
        
        visualize_trees_grid(
            display_trees,
            output_path=str(output_path / f"{prefix}_grid.png"),
            suptitle=f'Tree Structures (N={n_nodes}, H={max_depth}, Total={len(trees)})'
        )
    else:
        # 화면에 표시
        display_trees = trees[:min(len(trees), max_display)]
        fig = visualize_trees_grid(display_trees)
        if fig:
            plt.show()
