import { create } from 'zustand';

const useClaimsStore = create((set, get) => ({
  claims: [],
  currentClaim: null,
  loading: false,
  error: null,

  // HITL queue
  hitlQueue: [],
  hitlStats: null,
  hitlLoading: false,

  // Appeals
  appeals: [],
  appealLoading: false,

  // Filters
  statusFilter: 'all',
  sortBy: 'created_at',
  sortOrder: 'desc',

  // Actions
  setClaims: (claims) => set({ claims }),
  setCurrentClaim: (claim) => set({ currentClaim: claim }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),

  addClaim: (claim) => set((s) => ({ claims: [claim, ...s.claims] })),
  updateClaimStatus: (claimId, status) =>
    set((s) => ({
      claims: s.claims.map((c) => (c.claim_id === claimId ? { ...c, status } : c)),
    })),

  setHITLQueue: (queue) => set({ hitlQueue: queue }),
  setHITLStats: (stats) => set({ hitlStats: stats }),
  setHITLLoading: (loading) => set({ hitlLoading: loading }),
  removeFromHITL: (ticketId) =>
    set((s) => ({ hitlQueue: s.hitlQueue.filter((t) => t.ticket_id !== ticketId) })),

  setAppeals: (appeals) => set({ appeals }),
  setAppealLoading: (loading) => set({ appealLoading: loading }),

  setStatusFilter: (filter) => set({ statusFilter: filter }),
  setSortBy: (sortBy) => set({ sortBy }),

  getFilteredClaims: () => {
    const { claims, statusFilter, sortBy, sortOrder } = get();
    let filtered = statusFilter === 'all' ? claims : claims.filter((c) => c.status === statusFilter);
    filtered.sort((a, b) => {
      const aVal = a[sortBy] || '';
      const bVal = b[sortBy] || '';
      return sortOrder === 'desc' ? (bVal > aVal ? 1 : -1) : (aVal > bVal ? 1 : -1);
    });
    return filtered;
  },
}));

export default useClaimsStore;
