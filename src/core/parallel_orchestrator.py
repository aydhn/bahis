"""
parallel_orchestrator.py – Çok Çekirdekli (Multi-Core) İşlem Orkestratörü.

Python'un GIL kısıtlamasını aşmak için simülasyon ve optimizasyon gibi 
ağır matematiksel işlemleri farklı çekirdeklere dağıtır.
"""
import concurrent.futures
import multiprocessing
import asyncio
from loguru import logger
from typing import Callable, Any, List

class ParallelOrchestrator:
    def __init__(self):
        self.n_cores = multiprocessing.cpu_count()
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cores)
        logger.info(f"[Parallel] {self.n_cores} çekirdekli orkestratör hazır.")

    async def run_in_parallel(self, func: Callable, args_list: List[Any]):
        """Fonksiyonu argüman listesiyle çekirdeklere dağıtır."""
        loop = asyncio.get_running_loop()
        tasks = []
        for arg in args_list:
            tasks.append(loop.run_in_executor(self.executor, func, arg))
        
        results = await asyncio.gather(*tasks)
        return results

    def shutdown(self):
        self.executor.shutdown()
