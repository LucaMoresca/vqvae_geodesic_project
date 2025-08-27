# geodesic_kmedoids.py

import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class GeodesicKMedoids:
    """
    PAM geodetico su un grafo k-NN pesato con distanze euclidee locali.

    Parametri
    ---------
    G_csr : csr_matrix
        Grafo k-NN in formato CSR (N_lcc x N_lcc).
    Z_ref : np.ndarray
        Latenti corrispondenti alla LCC (N_lcc, d).
    k : int
        Numero di medoid (dimensione codebook).
    iter_max : int
        Iterazioni massime PAM.
    update_candidates : int
        Numero candidati da valutare per aggiornare ogni medoid.
    random_state : int
        Seed riproducibile.
    """

    def __init__(
        self,
        G_csr: csr_matrix,
        Z_ref: np.ndarray,
        k: int = 256,
        iter_max: int = 5,
        update_candidates: int = 64,
        random_state: int = 42,
    ):
        self.G = G_csr
        self.Z = Z_ref
        self.k = int(k)
        self.iter_max = int(iter_max)
        self.update_candidates = int(update_candidates)
        self.random_state = int(random_state)

        self.medoid_idx_: Optional[np.ndarray] = None     # (k,)
        self.assign_: Optional[np.ndarray] = None         # (N_lcc,)
        self.dist_to_medoid_: Optional[np.ndarray] = None # (N_lcc,)

    # ---------- utility private ----------
    def _geo_from_sources(self, sources: np.ndarray) -> np.ndarray:
        """Distanze geodetiche da ciascuna sorgente a TUTTI i nodi (stack di Dijkstra)."""
        D_list = []
        for s in sources:
            d = dijkstra(csgraph=self.G, directed=False,
                         indices=int(s), return_predecessors=False)
            D_list.append(d)
        return np.vstack(D_list)

    def _init_medoids_kpp(self) -> np.ndarray:
        """Init: KMeans++ in euclideo, poi mappa ogni centro al nodo più vicino → medoid iniziali."""
        km = KMeans(n_clusters=self.k, n_init=1,
                    init="k-means++", random_state=self.random_state)
        km.fit(self.Z)
        centers = km.cluster_centers_  # (k, d)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.Z)
        _, idxs = nn.kneighbors(centers, return_distance=True)
        return idxs.ravel()

    def _assign(self, medoids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assegna ogni nodo al medoid più vicino in geodetica."""
        D = self._geo_from_sources(medoids)                 # (k, N_lcc)
        assign = np.argmin(D, axis=0)                       # (N_lcc,)
        dist_min = D[assign, np.arange(D.shape[1])]         # (N_lcc,)
        return assign, dist_min

    def _update_medoids(self, medoids: np.ndarray, assign: np.ndarray) -> np.ndarray:
        """Update approssimato dei medoid di ciascun cluster."""
        new_meds = medoids.copy()
        rng = np.random.default_rng(self.random_state)

        for c in range(self.k):
            cluster_nodes = np.where(assign == c)[0]
            if cluster_nodes.size == 0:
                continue  # cluster vuoto

            current_med = medoids[c]

            # distanze dal medoid corrente
            d_med = dijkstra(csgraph=self.G, directed=False,
                             indices=int(current_med), return_predecessors=False)

            # candidati
            cand_sorted = cluster_nodes[np.argsort(d_med[cluster_nodes])]
            cand = cand_sorted[:min(self.update_candidates, cand_sorted.size)]
            if current_med not in cand:
                cand = np.concatenate([cand, np.array([current_med])], axis=0)

            # valutazione costo (1-median)
            MAX_CLUSTER_EVAL = 4000
            if cluster_nodes.size > MAX_CLUSTER_EVAL:
                eval_subset = rng.choice(cluster_nodes, size=MAX_CLUSTER_EVAL, replace=False)
            else:
                eval_subset = cluster_nodes

            best_med, best_cost = current_med, np.inf
            for u in cand:
                d_u = dijkstra(csgraph=self.G, directed=False,
                               indices=int(u), return_predecessors=False)
                cost = float(np.nansum(d_u[eval_subset]))
                if cost < best_cost:
                    best_cost, best_med = cost, int(u)

            new_meds[c] = best_med

        return new_meds

    # ---------- API ----------
    def fit(self):
        """Esegue PAM geodetico."""
        medoids = self._init_medoids_kpp()

        for it in range(1, self.iter_max + 1):
            assign, dist_min = self._assign(medoids)
            total_cost = float(np.nansum(dist_min))
            print(f"[Geodesic K-Medoids] iter {it}/{self.iter_max} | cost={total_cost:,.2f}")

            new_medoids = self._update_medoids(medoids, assign)

            if np.array_equal(new_medoids, medoids):
                print("Convergenza: medoid invariati.")
                break

            medoids = new_medoids

        # finalizzazione
        assign, dist_min = self._assign(medoids)
        self.medoid_idx_ = medoids
        self.assign_ = assign
        self.dist_to_medoid_ = dist_min

        return self

