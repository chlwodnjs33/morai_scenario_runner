try:
    from scenario_runner.utils.link_graph import LinkGraph
except ImportError:
    from utils.link_graph import LinkGraph


def build_simple_route(start_link: str, end_link: str):
    """
    예전 1차 테스트용.
    start/end만 넣기 때문에 실제 주행 route로는 부정확할 수 있음.
    """
    if start_link == end_link:
        return [start_link]
    return [start_link, end_link]


def build_route_between(map_loader, start_link: str, end_link: str):
    """
    link graph를 이용해 start_link에서 end_link까지 연결된 전체 link sequence 생성.
    """
    graph = LinkGraph(map_loader)
    route_links = graph.shortest_path(start_link, end_link)

    if route_links is None:
        raise RuntimeError(
            f"No connected route found: {start_link} -> {end_link}. "
            f"두 link가 같은 방향 경로로 연결되지 않았을 가능성이 큼."
        )

    length_m = graph.path_length(route_links)
    return route_links, length_m
