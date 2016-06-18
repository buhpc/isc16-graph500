#include "edgeList.h"

#include <iostream>
#include <cmath>

int main(int argc, char **argv) {
	// get scale and edge factor
    int scale = atoi(argv[1]);
    int edgeFactor = atoi(argv[2]);

    // compute num nodes & edges
    int numNodes = pow(2, scale);
    int numEdges = edgeFactor * numNodes;

    int testNodes = 10;
    int testEdges = 10;

    // allocate buffer 
    EdgeList edges(testEdges);

    // create edges
	edges.create(numNodes, scale);

	// prints Kronecker generated graph from and to vertices
    edge *edgeList = edges.edges();
    for (int i = 0; i < testEdges; i++) {
    	cout << edgeList[i].from << " " << edgeList[i].to << endl;
    }

	return 0;
}