# Adaptive Matrix Factorization For Graph Embedding
This is the official implementation for the paper titled with "Adaptive Matrix Factorization For Graph Embedding

### Motivation

two core issues about matrix factorization in the graph embedding domain remain addressed. First, given a concerned proximity matrix, the selected order of proximity should mirror the most representative information of a graph such that the resultant embedding can keep the maximal information. But how to cast the best order of proximity is still an open question. Second, factorization of a matrix is not unique and the ambiguity is rarely considered. For example, if one opts to use the singular value decomposition, how to optimally split the matrix of singular values remains an unresolved issue. Particularly when the matrix is symmetric positive definite (e.g., graph Laplacian), a spectral decomposition (which is equivalent to the singular value decomposition) allows raising fractional powers of the eigenvalues, resulting in a filtered matrix, whose factors may possess more expressive power than those of the matrix under an integral degree polynomial. Nonetheless, optimal fractional powers are unfortunately unknown. To address these issues and improve the effectiveness of matrix factorization for graph embedding, in this work we show that tuning orders of a proximity matrix and tuning powers of singular values are intrinsically connected. Then, we propose to raise fractional powers separately for each singular value, such that they are more adaptive to the relative importance of the corresponding singular subspaces when forming the node embedding.

### Requirements

Python 3.7



