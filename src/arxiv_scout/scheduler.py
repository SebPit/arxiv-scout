from apscheduler.schedulers.background import BackgroundScheduler

def start_scheduler(config_path: str, db_path: str, fetch_time: str = "06:00"):
    """Start background scheduler for daily pipeline runs (with email digest)."""
    from arxiv_scout.pipeline import run_pipeline

    hour, minute = map(int, fetch_time.split(":"))
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_pipeline, "cron", hour=hour, minute=minute,
        args=[config_path, db_path],
        kwargs={"send_email": True},
        id="daily_fetch",
        misfire_grace_time=3600,
    )
    scheduler.start()
    return scheduler
