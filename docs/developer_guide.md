# Developer Guide

## Principles

- Keep model integrations thin and isolated.
- Favor typed schemas over ad-hoc dicts.
- Route all planning decisions through orchestration interfaces.

## Extension Workflow

1. Add adapter class inheriting `CVPRModel`.
2. Register adapter in `ModelRegistry` and `configs/models.yaml`.
3. Add schema mappings and update tests.

## TODO

- TODO: Add coding standards and code owners.
