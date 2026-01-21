"""
트리 구조 시각화 스크립트
- b: base (루트)
- g: groove
- s: step
"""
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_trees(filepath):
    """JSON 파일에서 트리 데이터 로드"""
    with open(filepath, 'r') as f:
        return json.load(f)


def build_graph(tree_data):
    """트리 데이터를 networkx 그래프로 변환"""
    G = nx.DiGraph()
    
    # 노드 추가
    for node in tree_data['nodes']:
        G.add_node(node['id'], label=node['label'], depth=node['depth'])
    
    # 엣지 추가
    for edge in tree_data['edges']:
        G.add_edge(edge['source'], edge['target'])
    
    return G


def get_node_colors(G):
    """노드 타입에 따른 색상 반환"""
    color_map = {
        'b': '#FF6B6B',  # 빨간색 - base
        'g': '#4ECDC4',  # 청록색 - groove
        's': '#FFE66D'   # 노란색 - step
    }
    return [color_map[G.nodes[node]['label']] for node in G.nodes()]


def visualize_tree(G, tree_data, ax):
    """단일 트리 시각화"""
    # 계층적 레이아웃 계산
    pos = hierarchy_pos(G, 0)
    
    # 노드 색상
    colors = get_node_colors(G)
    
    # 노드 라벨
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    
    # 그래프 그리기
    nx.draw(G, pos, ax=ax, 
            node_color=colors,
            node_size=800,
            labels=labels,
            font_size=14,
            font_weight='bold',
            arrows=True,
            arrowsize=15,
            edge_color='#666666',
            width=2)
    
    ax.set_title(f"ID: {tree_data['id']}\n{tree_data['canonical']}", 
                 fontsize=10, fontweight='bold')


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """계층적 트리 레이아웃 계산"""
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, 
                   pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    
    children = list(G.successors(root))
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                pos=pos, parent=root, parsed=parsed)
    return pos


def main():
    # 데이터 로드
    trees = load_trees('trees_N6_H3.json')
    
    print(f"총 트리 개수: {len(trees)}")
    print(f"노드 수 (N): {trees[0]['N']}")
    print(f"최대 깊이 제약: {trees[0]['max_depth_constraint']}")
    print()
    
    # 트리 구조 분석
    label_counts = {'b': 0, 'g': 0, 's': 0}
    for tree in trees:
        for node in tree['nodes']:
            label_counts[node['label']] += 1
    
    print("전체 노드 라벨 분포:")
    print(f"  - b (base): {label_counts['b']} (각 트리당 1개 = 루트)")
    print(f"  - g (groove): {label_counts['g']}")
    print(f"  - s (step): {label_counts['s']}")
    print()
    
    # 선택된 트리 시각화 (다양한 구조 선택)
    selected_indices = [0, 4, 17, 37, 68, 89]  # 다양한 구조 선택
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, tree_idx in enumerate(selected_indices):
        tree = trees[tree_idx]
        G = build_graph(tree)
        visualize_tree(G, tree, axes[idx])
    
    # 범례 추가
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='b (base/root)'),
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label='g (groove)'),
        mpatches.Patch(facecolor='#FFE66D', edgecolor='black', label='s (step)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
               fontsize=12, bbox_to_anchor=(0.5, 0.98))
    
    plt.suptitle('Tree Structures (N=6, H=3)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('tree_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("시각화가 'tree_visualization.png'로 저장되었습니다.")


if __name__ == '__main__':
    main()


