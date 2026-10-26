# Run179 invalid shell redirection

The invoking shell failed to open `evidence/20260925_loop047_tail/run179/driver.log` because that directory did not yet exist. The runner never started. No borrowed source patch, service, benchmark, or NPU work occurred. The same experiment moves to Run180 after explicitly creating its evidence directory before redirection.
