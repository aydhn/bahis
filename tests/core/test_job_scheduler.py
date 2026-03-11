import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

import src.core.job_scheduler
from src.core.job_scheduler import JobScheduler

@pytest.fixture
def mock_apscheduler_ok():
    with patch("src.core.job_scheduler.APSCHEDULER_OK", True):
        # Even if installed, we mock the real scheduler so tests are fast and predictable
        with patch("src.core.job_scheduler.AsyncIOScheduler", create=True) as mock_scheduler_cls, \
                 patch("src.core.job_scheduler.MemoryJobStore", create=True), \
                 patch("src.core.job_scheduler.EVENT_JOB_EXECUTED", 1, create=True), \
                 patch("src.core.job_scheduler.EVENT_JOB_ERROR", 2, create=True) as mock_scheduler_cls:
            yield mock_scheduler_cls

@pytest.fixture
def mock_apscheduler_fail():
    with patch("src.core.job_scheduler.APSCHEDULER_OK", False):
        yield

@pytest.mark.asyncio
async def test_job_scheduler_fallback_add_jobs(mock_apscheduler_fail):
    scheduler = JobScheduler()

    def dummy_func(): pass

    scheduler.add_interval("test_int", dummy_func, seconds=5)
    assert len(scheduler._jobs) == 1
    assert scheduler._jobs[0]["name"] == "test_int"
    assert scheduler._jobs[0]["type"] == "interval"
    assert scheduler._jobs[0]["interval_seconds"] == 5

    scheduler.add_cron("test_cron", dummy_func, hour=12, minute=30)
    assert len(scheduler._jobs) == 2
    assert scheduler._jobs[1]["name"] == "test_cron"
    assert scheduler._jobs[1]["type"] == "cron"

    future_date = datetime.now() + timedelta(days=1)
    scheduler.add_date("test_date", dummy_func, run_date=future_date)
    assert len(scheduler._jobs) == 3
    assert scheduler._jobs[2]["name"] == "test_date"
    assert scheduler._jobs[2]["type"] == "date"

@pytest.mark.asyncio
async def test_job_scheduler_fallback_start_stop(mock_apscheduler_fail):
    scheduler = JobScheduler()

    executed = asyncio.Event()
    def dummy_func():
        executed.set()

    scheduler.add_interval("test_int", dummy_func, seconds=0)

    await scheduler.start()
    assert scheduler.is_running
    assert len(scheduler._fallback_tasks) == 1

    # Wait for the event to be set, timeout to prevent infinite loops
    try:
        await asyncio.wait_for(executed.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Task was not executed in time")

    await scheduler.stop()
    assert not scheduler.is_running

    # fallback_tasks cleared

@pytest.mark.asyncio
async def test_job_scheduler_fallback_date_execution(mock_apscheduler_fail):
    scheduler = JobScheduler()

    executed = asyncio.Event()
    def dummy_func():
        executed.set()

    past_date = datetime.now() - timedelta(seconds=1)
    scheduler.add_date("test_date", dummy_func, run_date=past_date)

    await scheduler.start()

    try:
        await asyncio.wait_for(executed.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Date task was not executed in time")

    await scheduler.stop()

@pytest.mark.asyncio
async def test_job_scheduler_apscheduler_ok(mock_apscheduler_ok):
    scheduler = JobScheduler()

    assert scheduler._scheduler is not None

    def dummy_func(): pass

    with patch("src.core.job_scheduler.IntervalTrigger", create=True) as mock_interval:
        scheduler.add_interval("test_int", dummy_func, seconds=10)
        mock_interval.assert_called_once_with(seconds=10, minutes=0, hours=0)
        scheduler._scheduler.add_job.assert_called()

    with patch("src.core.job_scheduler.CronTrigger", create=True) as mock_cron:
        scheduler.add_cron("test_cron", dummy_func, hour=10, minute=0)
        mock_cron.assert_called_once_with(hour=10, minute=0, day_of_week='mon-sun', timezone='Europe/Istanbul')

    future_date = datetime.now() + timedelta(days=1)
    with patch("src.core.job_scheduler.DateTrigger", create=True) as mock_date:
        scheduler.add_date("test_date", dummy_func, run_date=future_date)
        mock_date.assert_called_once_with(run_date=future_date, timezone='Europe/Istanbul')

    await scheduler.start()
    scheduler._scheduler.start.assert_called_once()

    await scheduler.stop()
    scheduler._scheduler.shutdown.assert_called_once_with(wait=False)

    scheduler._scheduler.running = True
    assert scheduler.is_running

@pytest.mark.asyncio
async def test_job_scheduler_apscheduler_get_remove_jobs(mock_apscheduler_ok):
    scheduler = JobScheduler()

    mock_job = MagicMock()
    mock_job.id = "1"
    mock_job.name = "test_job"
    mock_job.next_run_time = "2024-01-01"
    mock_job.trigger = "cron"

    scheduler._scheduler.get_jobs.return_value = [mock_job]

    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0]["id"] == "1"
    assert jobs[0]["name"] == "test_job"
    assert jobs[0]["next_run"] == "2024-01-01"

    scheduler.remove_job("test_job")
    scheduler._scheduler.remove_job.assert_called_once_with("test_job")

@pytest.mark.asyncio
async def test_job_scheduler_fallback_async_execution(mock_apscheduler_fail):
    scheduler = JobScheduler()

    executed = asyncio.Event()
    async def dummy_async_func():
        executed.set()

    scheduler.add_date("test_async_date", dummy_async_func, run_date=datetime.now() - timedelta(seconds=1))

    await scheduler.start()

    try:
        await asyncio.wait_for(executed.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Async task was not executed in time")

    await scheduler.stop()
