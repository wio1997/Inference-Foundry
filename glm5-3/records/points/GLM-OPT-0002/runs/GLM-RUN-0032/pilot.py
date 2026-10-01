import asyncio,sys
from pathlib import Path
from paired_wave import main
from owner_guard import start_watchdog
start_watchdog()
root=Path(__file__).parent
sys.exit(asyncio.run(main(root/"pilot",root.name,"active_count")))
