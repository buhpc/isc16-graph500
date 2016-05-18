# isc16-graph500

Graph500 is one of the applications for ISC16's student cluster competition. The Graph500 is a rating of supercomputer systems, focused on Data intensive loads. The intent of this Graph500 benchmark problem ("Search") is to develop a compact application that has multiple analysis techniques (multiple kernels) accessing a single data structure representing a weighted, undirected graph. In addition to a kernel to construct the graph from the input tuple list, there is one additional computational kernel to operate on the graph. This benchmark includes a scalable data generator which produces edge tuples containing the start vertex and end vertex for each edge.

The first kernel constructs an undirected graph in a format usable by all subsequent kernels. No subsequent modifications are permitted to benefit specific kernels. The second kernel performs a breadth-first search of the graph. Please note that teams are required to write their own implementation of Graph500 to run on a cluster; the details and reference implementation can be found at the web site: [www.graph500.org](http://graph500.org)

### Rules for Graph500
- The complete set of rules for ISC16's student cluster competition applications are outlined [here](http://www.hpcadvisorycouncil.com/events/2016/isc16-student-cluster-competition/benchmarking.php).
- Each team will need to implement with their own algorithm. It is not allowed to use the reference implementation for the runs; the implementation can be a simple or elaborate algorithm but each team will need to come up with its own original algorithm.
- The implementation must run across multiple nodes; the result of the runs that run on a single node only is not allowed.
- The edge factor should not be modified. The default value of edge factor should be 16.
- The scale factor should be used based on memory available for achieving the best results. Results from extremely low scale factors would not be considered. Please be aware that only runs that are conducted during the competition will be accepted.
- Please include a detailed description of your design. Although there is no hard requirement on the length of the description, the description should explain the algorithm being implemented, as well as the implementation details of your design. The description will be used by the judges to walk through the code during code inspection.
- The code will need to be submitted along with the results generated on the team’s own cluster. Please be sure to document and put adequate comments in code.
- The complete score for graph500 will take into account of everything that are being submitted. Such as the detailed description (on algorithm and implementation details), comments in the code, as well as the results.
- Please consider writing the code as portable as possible because the algorithm would potentially be analyzed or verified on a another cluster,
