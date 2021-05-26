# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:04:23 2021

@author: holge
"""

import asyncio
import time
from typing import List

from dask.distributed import Client, get_worker
from dask.utils import format_time


async def ping(addresses: List[str]):
    await asyncio.gather(*[
        get_worker().rpc(addr).identity()
        for addr in addresses
        if addr != get_worker().address
    ])


def all_ping(client: Client):
    workers = list(client.scheduler_info()["workers"])
    start = time.time()
    client.run(ping, workers)
    stop = time.time()
    print(format_time(stop - start))

cluster = 'localhost:8001'
if __name__ == "__main__":
    with Client(cluster) as client:
        all_ping(client)