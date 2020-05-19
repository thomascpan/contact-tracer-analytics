import pandas as pd
import numpy as np
import geopy.distance
from datetime import timedelta
import networkx as nx
import matplotlib.pyplot as plt
import operator

COLUMN_NAMES = ["datetime", "source", "target"]


def distance(lat1: str, long1: str, lat2: str, long2: str) -> float:
    """ Calculates distance between two coordinates
    Args:
        lat1 (str): latitude of first point
        long1 (str): longitude of first point
        lat2 (str): latitude of seocnd point
        long2 (str): longitude of second point
    Returns:
        float: distance between two points in feet
    """
    return geopy.distance.distance((lat1, long1), (lat2, long2)).feet


def withinDistance(row1: pd.core.series.Series, row2: pd.core.series.Series, threshold: int = 10) -> bool:
    """ Determines whether two buses were within distance of each other
    Args:
        row1 (pd.core.series.Series): record of a bus gps location
        row2 (pd.core.series.Series): record of a bus gps location
        threshold (int): distance threshold or max distance apart
    Returns:
        bool: whether buses where within distance of each other
    """
    return distance(row1.latitude, row1.longitude, row2.latitude, row2.longitude) <= threshold


def withinTime(row1: pd.core.series.Series, row2: pd.core.series.Series, threshold: int = 60) -> bool:
    """ Determines whether two buses were within a certain time period of each other
    Args:
        row1 (pd.core.series.Series): record of a bus gps location
        row2 (pd.core.series.Series): record of a bus gps location
        threshold (int): time threshold or max time apart
    Returns:
        bool: whether two buses were within a certain time period of each other
    """
    return abs((row1.datetime - row2.datetime).total_seconds()) <= threshold


def meanTimestamp(row1: pd.core.series.Series, row2: pd.core.series.Series) -> pd._libs.tslibs.timestamps.Timestamp:
    """ Gets the average of two timestamps
    Args:
        row1 (pd.core.series.Series): record of a bus gps location
        row2 (pd.core.series.Series): record of a bus gps location
    Returns:
        pd._libs.tslibs.timestamps.Timestamp: average timestamp of two bus records
    """
    return row1.datetime + (row2.datetime - row1.datetime) / 2


def sortPageRank(pr: dict, top: int = None) -> dict:
    """ Sort component based of "PageRank"
    Args:
        pr (dict): dict containing rank of each node
    Returns:
        dict: sorted dict based on rank
    """
    if not top:
        top = len(pr)
    return dict(sorted(pr.items(), key=operator.itemgetter(1), reverse=True)[:top])


def rank(pr: dict, query: set) -> dict:
    """ Get rank of connected component.
    Args:
        pr (dict): dict containing rank of each node
        query (set): set of nodes.
    Returns:
        dict: sorted nodes based on rank
    """
    results = dict((k, pr[k]) for k in query)
    return sortPageRank(results)


def preprocess(filepath: str, days: int = 7) -> pd.core.frame.DataFrame:
    """ Preprocess data.
    Args:
        filepath (str): path of file
        days (int): number of days to process
    Returns:
        pd.core.frame.DataFrame: new preprocessed dataframe object
    """
    rename = {"RecordedAtTime": "datetime", "VehicleRef": "id",
              "VehicleLocation.Latitude": "latitude", "VehicleLocation.Longitude": "longitude"}
    headers = list(rename.keys())
    df = pd.read_csv(filepath, usecols=headers, parse_dates=[headers[0]])
    df.rename(columns=rename, inplace=True)
    df.sort_values(by=["datetime", "latitude", "longitude"],
                   inplace=True, ignore_index=True)
    df.id = df.id.astype('category').cat.codes

    start_date = df.datetime.min().normalize()
    end_date = (start_date + timedelta(days=days)).normalize()

    mask = (df['datetime'] < end_date)
    df = df.loc[mask]

    temp = []
    cnt = len(df)

    for source_idx, source_row in df.iterrows():
        for target_idx in range(source_idx + 1, cnt):
            target_row = df.loc[target_idx]
            if (withinTime(source_row, target_row) and withinDistance(source_row, target_row)):
                if (source_row.id != target_row.id):
                    temp.append([meanTimestamp(source_row, target_row),
                                 source_row.id, target_row.id])
                    temp.append([meanTimestamp(target_row, source_row),
                                 target_row.id, source_row.id])
            else:
                break

    newDF = pd.DataFrame(temp, columns=COLUMN_NAMES)
    return newDF


def main():
    filepath = "mta_1706.csv"
    df = preprocess(filepath)
    # df.to_csv("final.csv", index=False, header=COLUMN_NAMES)

    G = nx.from_pandas_edgelist(df)
    pr = sortPageRank(nx.pagerank(G, 0.4))
    nx.draw_networkx(G)
    plt.show()
    cc = max(nx.connected_components(G), key=len)
    rankedCC = rank(pr, cc)

    print("rank\tid\tscore")
    for i, (k, v) in enumerate(rankedCC.items(), 1):
        print("%d\t%s\t%f" %(i, k, v))



if __name__ == "__main__":
    main()
