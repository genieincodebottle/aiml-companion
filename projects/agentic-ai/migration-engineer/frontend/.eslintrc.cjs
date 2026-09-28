/* ESLint config for the Autonomous Migration Engineer frontend (React 18 + Vite).
 * Formatting is owned by Prettier, so this focuses on correctness. Noisy
 * stylistic rules are downgraded to warnings so CI fails only on real problems. */
module.exports = {
  root: true,
  env: { browser: true, es2022: true, node: true },
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module', ecmaFeatures: { jsx: true } },
  settings: { react: { version: 'detect' } },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
  ],
  rules: {
    // Vite's automatic JSX runtime: React need not be in scope.
    'react/react-in-jsx-scope': 'off',
    'react/jsx-uses-react': 'off',
    // This codebase does not use prop-types by design.
    'react/prop-types': 'off',
    // Surface, do not block, on unused symbols.
    'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
  },
};
