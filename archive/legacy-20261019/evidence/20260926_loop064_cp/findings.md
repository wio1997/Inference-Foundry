# Loop064 findings

## Run287 and independent Bound Review

Original-path read-only census: 40/40 all-rank FULL Graph cohort reports, 60/60 exact1024 clients, 11,968 rank-cycles. Every rank has two owner requests, 16 full input update rows and 80 logical nonowner rows. Conservative compressed page envelopes do not overlap, including Astra High's separate same-domain owner compressed-history versus nonowner state-new-page check. Actual native scatter, recursive state, typed aliases, DSpark and prefix lifetime are not closed. Thus H002 remains a Resource candidate, without saved-byte or TPS estimate. Run285 leaves H001 numerical correctness inconclusive despite Run284 local join benefit. Astra High favors H003 hidden AllGather/local-Q overlap as the next shorter-closure Scheduling experiment; no ceiling or formal gain follows. See run287/findings.md and run287/astra_review.md.
