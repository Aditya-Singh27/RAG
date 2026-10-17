# Pydantic V2 Architecture and Performance

Pydantic V2 represents a complete rewrite of Python's most popular data validation library, achieving 5x to 50x performance improvements.

## Rust-Powered pydantic-core
The core validation engine was rewritten in Rust (`pydantic-core`). Python type hints are compiled at model declaration time into a static validation graph in Rust.
During data parsing:
1. Python objects (e.g. dicts, JSON strings) are passed directly into compiled Rust code.
2. Rust traverses the validation graph, coercing types and executing validators without Python interpreter overhead.
3. Errors are collected and formatted in a single pass.

## Serialization and JSON Handling
Pydantic V2 includes native JSON parsing and serialization powered by the `jiter` Rust crate. Calling `model.model_dump_json()` parses directly from C-level buffers, bypassing Python's intermediate dictionary creation. This dramatically lowers memory allocation overhead for high-throughput APIs.

## BaseModel Lifecycle
Models declare fields using standard Python type annotations. Validators are registered via `@field_validator` (for single attributes) and `@model_validator` (for cross-field validation, with `mode='before'` or `mode='after'`).
