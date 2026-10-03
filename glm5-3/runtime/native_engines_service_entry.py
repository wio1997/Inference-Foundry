"""Run the existing GLM V11 entry with independent native engine provenance."""
import service_entry
from native_engines_service_config import checked_config

service_entry.checked_config = checked_config

if __name__ == "__main__":
    service_entry.main()
