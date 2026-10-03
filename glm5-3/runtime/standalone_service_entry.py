"""Run the existing native-protocol gateway for one standalone engine."""
import service_entry
from standalone_service_config import checked_config

if __name__ == "__main__":
    service_entry.checked_config = checked_config
    service_entry.main()
