# run_tests.sh
#!/bin/bash
PYTHONPATH= pytest tests/ -v "$@"
