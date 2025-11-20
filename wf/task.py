from latch.resources.tasks import small_gpu_task


@small_gpu_task
def glue_task(
    string: str
) -> str:
    import time
    time.sleep(3600)
    return str
