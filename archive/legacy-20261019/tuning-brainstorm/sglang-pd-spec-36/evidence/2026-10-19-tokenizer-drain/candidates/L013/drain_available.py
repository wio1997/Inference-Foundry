"""UNTESTED CPU candidate. Not installed, deployed, or performance accepted.

Override collection only; the actual base class keeps its tokenizer execution,
kwargs fallback, future completion, cancellation checks, and exception delivery.
Unlike the baseline, this candidate never waits for a future queue arrival after
the first request. Arrival-dependent batch splitting is a required counterexample.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)
BASE_SOURCE_COMMIT = "ce0a8fea5f4245839d174b2b8c55692be1810a7f"
BASE_BATCHER_SHA256 = "a52ed01045e146dbddcd915f3ea1c0d15775207bebf26777df84bbaabb709ff6"


async def drain_dynamic_batch_loop(self):
    while True:
        try:
            prompt, kwargs, result_future = await self._queue.get()
            prompts = [prompt]
            kwargs_list = [kwargs]
            result_futures = [result_future]
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            while len(prompts) < self.max_batch_size:
                if loop.time() - start_time >= self.batch_wait_timeout_s:
                    break
                try:
                    prompt, kwargs, result_future = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                prompts.append(prompt)
                kwargs_list.append(kwargs)
                result_futures.append(result_future)
            logger.debug(
                f"AsyncDynamicbatchTokenizer: Processing dynamic batch of size {len(prompts)}"
            )
            await self._process_dynamic_batch(prompts, kwargs_list, result_futures)
        except Exception as exc:
            logger.error(f"Error in dynamic batch loop: {exc}")


def make_candidate(base_class):
    """The harness must verify the base module hash before passing its class."""
    class DrainAvailableTokenizer(base_class):
        _dynamic_batch_loop = drain_dynamic_batch_loop

    return DrainAvailableTokenizer
