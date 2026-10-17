# Python AsyncIO Internal Architecture

Python's `asyncio` module provides single-threaded cooperative multitasking using an event loop, coroutines, and futures.

## The Event Loop Core
At the center of asyncio is the `AbstractEventLoop`. In Python 3.8+ on Windows, the default event loop implementation is `ProactorEventLoop` (backed by I/O Completion Ports - IOCP), whereas Unix systems default to `SelectorEventLoop` (backed by epoll or kqueue).
The event loop continuously monitors file descriptors, timers, and scheduled callbacks. When an `await` expression is encountered on a coroutine, control is yielded back to the event loop, allowing other tasks to execute.

## Tasks vs Futures vs Coroutines
- A **Coroutine** is a generator-like function declared with `async def`. It produces a coroutine object that cannot run by itself until driven by an event loop or awaited.
- A **Future** represents an eventual result of an asynchronous operation. It has states: PENDING, CANCELLED, or FINISHED.
- A **Task** is a subclass of Future that wraps a coroutine and registers it with the event loop for execution. `asyncio.create_task()` schedules concurrent execution.

## ThreadPoolExecutor Integration
CPU-bound tasks or blocking synchronous I/O (such as legacy filesystem reads or heavy crypto) will freeze the event loop. The recommended pattern is offloading blocking functions using `loop.run_in_executor(None, sync_func, *args)` or `asyncio.to_thread(sync_func, *args)`.
