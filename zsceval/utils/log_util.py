import datetime


def eta(start_time: float, end_time: float, total: int, processed: int) -> str:
    """Compute the estimated time of arrival.

    Args:
        start_time (float): Start time.
        end_time (float): End time.
        total (int): Total number of items.
        processed (int): Number of processed items.

    Returns:
        str: Estimated time of arrival.
    """
    elapsed = end_time - start_time
    eta = (total - processed) * elapsed / processed
    return str(datetime.timedelta(seconds=int(eta)))


def get_table_str(items: list, headers: list = None, title: str = None, sort: bool = True) -> str:
    """
    return a table str
    """
    import io

    from rich.console import Console
    from rich.table import Table

    if headers:
        assert len(headers) == len(items[0])
        table = Table(title=title)
        for h in headers:
            table.add_column(h)
    else:
        table = Table(show_header=False, title=title)
        for _ in range(len(items[0])):
            table.add_column(justify="left")

    if sort:
        items = sorted(items, key=lambda x: x[0])

    for v in items:
        table.add_row(*[str(_v) for _v in v])
    string_io = io.StringIO()
    console = Console(file=string_io, record=True)
    console.print(table)
    return console.export_text()
