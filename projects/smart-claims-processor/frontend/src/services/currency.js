/**
 * Country-aware currency formatting.
 *
 * On app load, App.jsx calls `loadCurrency()` which fetches the active
 * country's currency symbol from the backend. All components use `fmt()`
 * instead of hardcoding "$".
 */
import api from './api';

let _symbol = '$';
let _code = 'USD';
let _locale = undefined;

const LOCALE_MAP = {
  us: 'en-US',
  india: 'en-IN',
};

/** Call once on app mount - fetches from GET /api/settings/country. */
export async function loadCurrency() {
  try {
    const res = await api.get('/api/settings/country');
    _symbol = res.data?.active?.currency_symbol || '$';
    _code = res.data?.active?.currency || 'USD';
    const countryCode = res.data?.active?.code?.toLowerCase();
    _locale = LOCALE_MAP[countryCode] || undefined;
  } catch {
    // Fallback to $ if backend isn't reachable yet (login page).
  }
}

export function currencySymbol() { return _symbol; }
export function currencyCode() { return _code; }

/** Format a numeric value as a currency string. Returns '-' for null/undefined. */
export function fmt(value) {
  if (value == null || value === '') return '-';
  return `${_symbol}${parseFloat(value).toLocaleString(_locale, { minimumFractionDigits: 0, maximumFractionDigits: 2 })}`;
}
