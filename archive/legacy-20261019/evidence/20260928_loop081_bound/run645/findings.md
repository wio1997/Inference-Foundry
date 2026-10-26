# Run645 inter-Graph native coverage

SHA-pinned all8 Run611 Level0 task_time, two adjacent Model45 Graph transitions. Rank7 windows24.591/22.214ms; union of **all exported native tasks** leaves16.267/13.941ms cumulatively uncovered. Physical kernel/copy union is8.103/8.080ms. The uncovered intervals are **distributed** across258/253 positive gaps; largest single gap0.863/0.380ms. This is not one large Host stall or certified hardware idle.

Stream47 physical task sums on rank7 are7.119/7.112ms, similar to other ranks (~7ms). Stream38 carries `aiv_all_gather/reduce_scatter/all_to_all` named tasks: rank7 cumulative0.929/0.913ms versus earlier ranks roughly5–7ms, consistent with peer waiting on earlier ranks. Same task count per rank does not make duration intrinsic service. Astra independently recomputed all16 windows and recommends exact Host flow/queue lineage before any overlap intervention. No removable time or whole-Product TPS follows.
