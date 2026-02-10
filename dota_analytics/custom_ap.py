import numpy as np

class CustomAffinityPropagation:
    """
    Implémentation custom de l'Affinity Propagation (Frey & Dueck).
    Reproduit le comportement de sklearn mais permet de voir le code.
    """
    def __init__(self, damping=0.9, max_iter=200, convergence_iter=15, verbose=True):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.verbose = verbose
        
        self.cluster_centers_indices_ = None
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, S):
        """
        S: Matrice de similarité (N, N).
        """
        N = S.shape[0]
        
        # R = Responsabilité, A = Disponibilité
        R = np.zeros((N, N))
        A = np.zeros((N, N))
        
        # Historique de convergence
        e = np.zeros((N, self.convergence_iter))
        
        for i in range(self.max_iter):
            self.n_iter_ = i
            
            # --- 1. Mise à jour Responsabilité (R) ---
            AS = A + S
            I = np.argmax(AS, axis=1)
            Y = AS[np.arange(N), I]
            
            AS[np.arange(N), I] = -np.inf
            Y2 = np.max(AS, axis=1)
            AS[np.arange(N), I] = Y
            
            max_AS = np.tile(Y, (N, 1)).T
            max_AS[np.arange(N), I] = Y2
            
            R_new = S - max_AS
            R = (1 - self.damping) * R_new + self.damping * R
            
            # --- 2. Mise à jour Disponibilité (A) ---
            Rp = np.maximum(0, R)
            Rp[np.arange(N), np.arange(N)] = 0
            sum_Rp = np.sum(Rp, axis=0)
            
            diag_R = np.diag(R)
            A_new = np.tile(sum_Rp + diag_R, (N, 1)) - Rp
            A_new = np.minimum(0, A_new)
            A_new[np.arange(N), np.arange(N)] = sum_Rp
            
            A = (1 - self.damping) * A_new + self.damping * A
            
            # --- 3. Convergence ---
            E = (np.diag(A) + np.diag(R)) > 0
            e[:, i % self.convergence_iter] = E
            K = np.sum(E)
            
            if i >= self.convergence_iter:
                se = np.sum(e, axis=1)
                unconverged = np.sum((se == self.convergence_iter) + (se == 0)) != N
                if (not unconverged) and (K > 0):
                    if self.verbose:
                        print(f"🚀 Convergence atteinte à l'itération {i}. {K} clusters trouvés.")
                    break
        else:
            if self.verbose:
                print("⚠️ Max itérations atteint avant convergence.")

        # --- 4. Résultats ---
        I = np.where(np.diag(A + R) > 0)[0]
        self.cluster_centers_indices_ = I
        
        if len(I) > 0:
            c = np.argmax(S[:, I], axis=1)
            self.labels_ = I[c]
            
            # Remapping des labels 0, 1, 2...
            unique_labels = np.unique(self.labels_)
            mapper = {old: new for new, old in enumerate(unique_labels)}
            self.labels_ = np.array([mapper[l] for l in self.labels_])
        else:
            self.labels_ = np.array([-1] * N)

        return self